import torch
class Config:
    # Data
    dataset_name = "ParsBench/parsinlu-machine-translation-fa-en-alpaca-style"
    source_lang = "instruction"   # English
    target_lang = "output"        # Persian
    max_length = 32
    batch_size = 24

    # Model
    input_dim = 5000             # Vocabulary size for English
    output_dim = 5000            # Vocabulary size for Persian 
    embedding_dim = 64           # Word vector dimensions
    hidden_dim = 128              # LSTM hidden state size
    num_layers = 1                # Stacked LSTM layers
    dropout = 0.1                 # Regularization to prevent overfitting

    # Training
    learning_rate = 0.001
    num_epochs = 5
    teacher_forcing_ratio = 0.7   # Mix of ground truth vs model predictions


    # Optimization
    gradient_accumulation_steps = 1
    use_amp = True                 # Mixed precision for speed
    use_gradient_clipping = True
    max_grad_norm = 1.0


    # Checkpoint Configuration =====
    checkpoint_interval = 1        # Save every 2 epochs
    save_best_only = True          # Only save when model improves

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    model_save_path = "models/seq2seq_model.pth"
    tokenizer_save_path = "models/tokenizers/"
    checkpoint_path = "models/checkpoint.pth"
    best_model_path = "models/best_model.pth"