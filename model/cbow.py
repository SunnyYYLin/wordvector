import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CBOW(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 emb_dim, 
                 shared_embeddings=False):
        super(CBOW, self).__init__()
        self.bag_emb = nn.Embedding(vocab_size, emb_dim)
        self.target_emb = nn.Embedding(vocab_size, emb_dim)
        if shared_embeddings:
            self.bag_emb.weight = self.target_emb.weight
        self.emb_dim = emb_dim

    def forward(self, bags: torch.Tensor):
        """
        bags: (batch_size, window_size - 1)
        """
        bags_emb_avg = self.bag_emb(bags).sum(dim=1) # (batch_size, emb_dim)
        probs = F.log_softmax(torch.mm(bags_emb_avg, self.target_emb.weight.t()), dim=1) # (batch_size, vocab_size)
        return probs
    
    @staticmethod
    def loss(probs: torch.Tensor, targets: torch.Tensor, negatives: torch.Tensor):
        """
        probs: (batch_size, vocab_size)
        targets: (batch_size,)
        negatives: (batch_size, neg_size)
        """
        pos = probs.gather(1, targets.view(-1, 1))  # (batch_size, 1)
        neg = probs.gather(1, negatives)  # (batch_size, neg_size)
        pos_loss = -pos.squeeze()  # (batch_size,)
        neg_loss = neg.exp().sum(dim=1).log()  # (batch_size,)
        loss = pos_loss + neg_loss
        return loss.mean()
