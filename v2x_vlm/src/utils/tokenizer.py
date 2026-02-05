import torch

class TrajectoryTokenizer:
    def __init__(self, cfg):
        self.vocab_size = cfg.model.vocab_size
        self.min_val = cfg.model.min_coord
        self.max_val = cfg.model.max_coord

    def coords_to_tokens(self, coords):
        norm = (coords - self.min_val) / (self.max_val - self.min_val)
        tokens = (norm * self.vocab_size).long()
        return torch.clamp(tokens, 0, self.vocab_size - 1)

    def tokens_to_coords(self, tokens):
        norm = (tokens.float() + 0.5) / self.vocab_size
        return norm * (self.max_val - self.min_val) + self.min_val