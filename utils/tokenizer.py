from collections import Counter

class Vocab:
    def __init__(self, counter, min_freq=2, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        self.itos = list(specials)                                           # index to string
        self.stoi = {value: key for key, value in enumerate(self.itos)}      # string to index

        for token, freq in counter.items():
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def __len__(self):
        return len(self.itos)                                   # Vocabulary size

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])         # Returns <unk> for unknown tokens

    def get_itos(self):
        return self.itos                                        # Get full vocabulary list             


def simple_tokenizer(text: str):
    return text.lower().strip().split()


def build_vocab(dataset, lang, min_freq=2):
    counter = Counter()
    for ex in dataset["train"]:
        if "translation" in ex:
            text = ex["translation"][lang]
        else:
            text = ex[lang]

        tokens = simple_tokenizer(text)
        counter.update(tokens)

    vocab = Vocab(counter, min_freq=min_freq)
    return simple_tokenizer, vocab