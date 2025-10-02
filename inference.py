import torch
from config import Config
from models.seq2seq import Encoder, Decoder, Seq2Seq
from utils.tokenizer import build_vocab
from datasets import load_from_disk

def translate_sentence(sentence, model, src_tokenizer, src_vocab, trg_vocab, device, max_len=30):
    model.eval()
    tokens = src_tokenizer(sentence.lower())
    src_tensor = torch.tensor([src_vocab["<sos>"]] + [src_vocab[t] for t in tokens] + [src_vocab["<eos>"]]).unsqueeze(1).to(device)
    with torch.no_grad():
        hidden = model.encoder(src_tensor)
    trg_indexes = [trg_vocab["<sos>"]]

    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab["<eos>"]:
            break
    return [trg_vocab.get_itos()[i] for i in trg_indexes]

if __name__ == "__main__":
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_from_disk("data/raw/")
    src_tokenizer, src_vocab = build_vocab(dataset, cfg.source_lang)
    trg_tokenizer, trg_vocab = build_vocab(dataset, cfg.target_lang)

    enc = Encoder(len(src_vocab), cfg.emb_dim, cfg.hid_dim, cfg.n_layers)
    dec = Decoder(len(trg_vocab), cfg.emb_dim, cfg.hid_dim, cfg.n_layers)
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))

    print(translate_sentence("I love cats", model, src_tokenizer, src_vocab, trg_vocab, device))
