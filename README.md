
🌐 Seq2Seq Translator

A Sequence-to-Sequence (Seq2Seq) Neural Machine Translation project built with PyTorch.
This repository implements a complete training and inference pipeline for translating text between languages (e.g., English ⟷ Persian) using encoder-decoder architecture.

📂 Project Structure
```bash
seq2seq_translator/
│
├── app.py                # Web app for translation (Gradio)
├── config.py             # Configuration settings
├── inference.py          # Run inference with trained models
├── resume_training.py    # Continue training from a checkpoint
├── train.py              # Training script
├── requirements.txt      # Python dependencies
│
├── data/
│   ├── download_data.py  # Download & prepare dataset
│   ├── processed/        # Preprocessed data
│   └── raw/              # Raw dataset files (train, test, validation)
│
├── models/
│   ├── seq2seq.py        # Seq2Seq model (encoder-decoder architecture)
│   └── tokenizers/       # Tokenizer files
│
└── utils/
    ├── preprocessing.py  # Preprocessing utilities
    └── tokenizer.py      # Vocabulary & tokenization
```

---

## 🚀 Features
- End-to-end **Seq2Seq translation pipeline** (training → evaluation → inference).  
- **Custom tokenizer and preprocessing** for bilingual datasets.  
- **Training script with checkpointing** (`train.py`, `resume_training.py`).  
- **Inference pipeline** for testing translations on new text.  
- **Interactive web app** (`app.py`) for user-friendly translation.  

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/seq2seq_translator.git
cd seq2seq_translator/seq2seq_translator
```

Create a virtual environment & install dependencies:

```bash
pip install -r requirements.txt
```

📊 Usage
1. Download Dataset
```bash
python data/download_data.py
```


2. Train Model
```bash
python train.py
```


3. Resume Training(If for any reason the train process is interrupted)
```bash
python resume_training.py
```


4. Run Inference
```bash
python inference.py
```


5. Launch Web App
```bash
python app.py
```


📈 Model
The model is a Seq2Seq Encoder-Decoder architecture implemented in models/seq2seq.py with attention mechanism support.

