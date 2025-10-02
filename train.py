import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
import os
from config import Config
from utils.tokenizer import build_vocab
from utils.preprocessing import collate_fn
from models.seq2seq import Encoder, Decoder, Seq2Seq
from tqdm import tqdm

def save_checkpoint(epoch, model, optimizer, scaler, loss, path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, scaler, path, device):
    """Load training checkpoint"""
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, best_loss
    return 0, float('inf')  # Start from beginning if no checkpoint

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch, cfg):
    model.train()
    total_loss = 0
    optimizer.zero_grad()  # Zero gradients at start
    
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for batch_idx, (src, trg) in enumerate(loop):
        src, trg = src.to(device), trg.to(device)

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg) / cfg.gradient_accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
            if cfg.use_gradient_clipping:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.gradient_accumulation_steps
        loop.set_postfix(loss=loss.item() * cfg.gradient_accumulation_steps)

    return total_loss / len(dataloader)



def main():
    cfg = Config()
    device = cfg.device
    print(f"Using device: {device}")

    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/tokenizers", exist_ok=True)

    # Load dataset (full dataset)
    dataset = load_from_disk("data/raw/")

    # Build vocab using full dataset
    src_tokenizer, src_vocab = build_vocab(dataset, cfg.source_lang)
    trg_tokenizer, trg_vocab = build_vocab(dataset, cfg.target_lang)

    # Save tokenizers and vocab for future use
    torch.save({
        'src_tokenizer': src_tokenizer,
        'src_vocab': src_vocab,
        'trg_tokenizer': trg_tokenizer,
        'trg_vocab': trg_vocab
    }, cfg.tokenizer_save_path + "tokenizers.pth")

    # DataLoader with train split
    collate = lambda batch: collate_fn(
        batch, src_tokenizer, trg_tokenizer, src_vocab, trg_vocab, cfg.max_length,
        src_lang=cfg.source_lang, trg_lang=cfg.target_lang
    )
    dataloader = DataLoader(dataset["train"], batch_size=cfg.batch_size, collate_fn=collate, shuffle=True)

    # Model
    enc = Encoder(len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.num_layers)
    dec = Decoder(len(trg_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.num_layers)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])
    scaler = torch.cuda.amp.GradScaler()

    # Try to load checkpoint
    start_epoch, best_loss = load_checkpoint(model, optimizer, scaler, cfg.checkpoint_path, device)

    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        
        try:
            loss = train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch, cfg)
            print(f"Epoch {epoch+1}/{cfg.num_epochs} | Loss: {loss:.3f}")

            # Save checkpoint after each epoch
            save_checkpoint(epoch, model, optimizer, scaler, loss, cfg.checkpoint_path)

            # Save best model
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), cfg.best_model_path)
                print(f"🎉 New best model saved with loss: {loss:.3f}")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("⚠️ GPU out of memory. Saving checkpoint and exiting...")
                save_checkpoint(epoch, model, optimizer, scaler, loss, cfg.checkpoint_path)
                print("✅ Checkpoint saved. You can resume training later.")
                break
            else:
                raise e

    print("✅ Training completed!")

if __name__ == "__main__":
    main()