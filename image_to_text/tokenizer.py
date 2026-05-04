from config import Config

class LaTeXTokenizer:
    def __init__(self):
        self.vocab = Config.special_tokens + Config.chars
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.sos_id = self.stoi[Config.sos_token]
        self.eos_id = self.stoi[Config.eos_token]
        self.pad_id = self.stoi[Config.pad_token]
        self.unk_id = self.stoi[Config.unk_token]

    def __len__(self):
        return len(self.vocab)

    def encode(self, formula, max_len=None):
        tokens = [Config.sos_token] + list(formula) + [Config.eos_token]
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if max_len:
            if len(ids) > max_len:
                ids = ids[:max_len-1] + [self.eos_id]
            else:
                ids += [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids, skip_special=True):
        chars = []
        for i in ids:
            if i == self.eos_id:
                break
            if i == self.pad_id:
                continue
            if skip_special and i in [self.sos_id, self.unk_id]:
                continue
            chars.append(self.itos[i])
        return ''.join(chars)