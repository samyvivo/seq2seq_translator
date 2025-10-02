import streamlit as st
import torch
from config import Config
from models.seq2seq import Encoder, Decoder, Seq2Seq
from utils.tokenizer import build_vocab
from datasets import load_from_disk

# -------------------
# Load Model + Vocabs
# -------------------
@st.cache_resource
def load_model_and_vocab():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_from_disk("data/raw/")
    src_tokenizer, src_vocab = build_vocab(dataset, cfg.source_lang)
    trg_tokenizer, trg_vocab = build_vocab(dataset, cfg.target_lang)

    enc = Encoder(len(src_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.num_layers)
    dec = Decoder(len(trg_vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.num_layers)
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    model.eval()

    return model, src_tokenizer, src_vocab, trg_vocab, device


def translate_sentence(sentence, model, src_tokenizer, src_vocab, trg_vocab, device, max_len=30):
    model.eval()
    tokens = src_tokenizer(sentence.lower())
    src_tensor = torch.tensor(
        [src_vocab["<sos>"]] + [src_vocab[t] for t in tokens] + [src_vocab["<eos>"]]
    ).unsqueeze(1).to(device)

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

    return [trg_vocab.get_itos()[i] for i in trg_indexes][1:-1]  # remove <sos>, <eos>


# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="English → Persian Translator", layout="centered")

st.title("📝 English → Persian Translator")
st.write("Type an English sentence and translate it into Persian using a Seq2Seq RNN model.")

model, src_tokenizer, src_vocab, trg_vocab, device = load_model_and_vocab()

user_input = st.text_area("Enter English text:", "I love cats")

if st.button("Translate"):
    with st.spinner("Translating..."):
        translation = translate_sentence(user_input, model, src_tokenizer, src_vocab, trg_vocab, device)
        st.success(" ".join(translation))
