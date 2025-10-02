import torch


def numericalize(tokens, vocab, max_len):
    ids = [vocab["<sos>"]] + [vocab[token] for token in tokens[:max_len - 2]] + [vocab["<eos>"]]
    return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch, src_tokenizer, trg_tokenizer, src_vocab, trg_vocab, max_len, src_lang="en", trg_lang="fa"):
    src_batch, trg_batch = [], []
    for ex in batch:
        # Detect structure
        if "translation" in ex:
            src_text = ex["translation"][src_lang]          # Nested structure
            trg_text = ex["translation"][trg_lang]
        else:  
            src_text = ex[src_lang]                         # Flat structure
            trg_text = ex[trg_lang]

        src_tokens = src_tokenizer(src_text)                # Tokenize English
        trg_tokens = trg_tokenizer(trg_text)                # Tokenize Persian

        # Add <sos> and <eos>
        src_indices = [src_vocab["<sos>"]] + [src_vocab[t] for t in src_tokens] + [src_vocab["<eos>"]]
        trg_indices = [trg_vocab["<sos>"]] + [trg_vocab[t] for t in trg_tokens] + [trg_vocab["<eos>"]]

        # Pad
        src_indices = src_indices[:max_len] + [src_vocab["<pad>"]] * (max_len - len(src_indices))
        trg_indices = trg_indices[:max_len] + [trg_vocab["<pad>"]] * (max_len - len(trg_indices))

        src_batch.append(src_indices)
        trg_batch.append(trg_indices)

    return torch.tensor(src_batch), torch.tensor(trg_batch)