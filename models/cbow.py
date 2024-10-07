import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    """Continuous Bag of Words (CBOW) model for word embeddings.
    This model learns word embeddings by predicting a tag word from a bag of context words.
        vocab_size (int): Size of the vocabulary.
        emb_dim (int): Dimensionality of the embeddings.
        shared_embeddings (bool, optional): If True, the embeddings for the context and tag words are shared. Default is False.
    Methods:
        average(bags: torch.Tensor) -> torch.Tensor:
            Computes the average embedding for a batch of bags of context words.
        forward(bags: torch.Tensor) -> torch.Tensor:
            Computes the probability distribution over the vocabulary for a batch of bags of context words.
        predict(bags: torch.Tensor) -> torch.Tensor:
            Predicts the most likely tag word for a batch of bags of context words.
        loss(pair, use_logsigmoid=True) -> torch.Tensor:
            Computes the loss for a batch of (context, tag, negative) triplets.
        _pos_loss(avg: torch.Tensor, tags: torch.Tensor, use_logsigmoid=True) -> torch.Tensor:
            Computes the positive loss for a batch of tag words.
        _neg_loss(avg: torch.Tensor, negatives: torch.Tensor, use_logsigmoid=True) -> torch.Tensor:
            Computes the negative loss for a batch of negative samples.
        nearest(word: torch.Tensor, k=10) -> list[torch.Tensor]:
            Finds the k nearest words to the given word in the embedding space.
    """
    def __init__(self, vocab_size, emb_dim, shared_embeddings=False):
        """
        """
        super(CBOW, self).__init__()
        self.bag_emb = nn.Embedding(vocab_size, emb_dim)
        self.tag_emb = nn.Embedding(vocab_size, emb_dim)
        if shared_embeddings:
            self.bag_emb.weight = self.tag_emb.weight
        self.emb_dim = emb_dim

    def average(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Compute the average embedding for each bag of words.
        Args:
            bags (torch.Tensor): A tensor of shape (batch_size, 2*window_size) containing indices of words in each bag.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, emb_dim) containing the average embedding for each bag.
        """
        bags_emb_avg = self.bag_emb(bags).mean(dim=1) # (batch_size, emb_dim)
        return bags_emb_avg

    def forward(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the CBOW model.
        Args:
            bags (torch.Tensor): A tensor of shape (batch_size, 2*window_size) containing indices of words in each bag.
        Returns:
            torch.Tensor : A tensor of shape (batch_size, vocab_size) containing the probability distribution over the vocabulary.
        """
        bags_emb_avg = self.average(bags)
        probs = torch.mm(bags_emb_avg, self.tag_emb.weight.t()) # (batch_size, vocab_size)
        return probs
    
    @torch.no_grad()
    def predict(self, bags: torch.Tensor) -> torch.Tensor:
        """
        Predict the class labels for the given input bags of words.
        Args:
            bags (torch.Tensor): A tensor of shape (batch_size, 2*window_size) containing indices of words in each bag.
        Returns:
            torch.Tensor: A tensor of shape (batch_size,) containing the predicted class labels.
        """
        probs = self.forward(bags)
        return torch.argmax(probs, dim=1)
    
    def loss(self, pair, use_logsigmoid=True) -> torch.Tensor:
        """
        Compute the loss for the given pair of bags, tags, and negatives.
        Args:
            pair (object): An object containing the following attributes:
            - bags (torch.Tensor): A tensor of shape (batch_size, 2*window_size, emb_dim) representing the context word embeddings.
            - tags (torch.Tensor): A tensor of shape (batch_size, emb_dim) representing the tag word embeddings.
            - negatives (torch.Tensor): A tensor of shape (batch_size, neg_size, emb_dim) representing the negative word embeddings.
            use_logsigmoid (bool, optional): If True, use the log-sigmoid function for loss computation. Defaults to True.
        Returns:
            torch.Tensor: A scalar tensor representing the mean loss.
        """
        
        bags, tags, negatives = pair.bags, pair.tags, pair.negatives
        avg = self.average(bags)

        pos_loss = self._pos_loss(avg, tags, use_logsigmoid)
        neg_loss = self._neg_loss(avg, negatives, use_logsigmoid)
        loss = pos_loss + neg_loss
        return loss.mean()
        
    def _pos_loss(self, avg: torch.Tensor, tags: torch.Tensor, use_logsigmoid=True) -> torch.Tensor:
        tag_emb = self.tag_emb(tags) # (batch_size, emb_dim)
        pos_loss = (tag_emb * avg).sum(dim=1) # (batch_size)
        if use_logsigmoid:
            pos_loss = F.logsigmoid(pos_loss)
        return -pos_loss
    
    def _neg_loss(self, avg: torch.Tensor, negatives: torch.Tensor, use_logsigmoid=True) -> torch.Tensor:
        neg_emb = self.tag_emb(negatives) # (batch_size, num_negatives, emb_dim)
        neg_loss = torch.bmm(neg_emb, avg.unsqueeze(2)).squeeze()  # (batch_size, num_negatives)
        if use_logsigmoid:
            neg_loss = -F.logsigmoid(-neg_loss.sum(dim=1))  # (batch_size)
        else:
            neg_loss = torch.exp(-neg_loss).sum(dim=1).log() # 直接计算
        return neg_loss
    
    def nearest(self, word_idx: int, k=10) -> list[int]:
        """
        Find the nearest words to a given word index based on cosine similarity.
        Args:
            word_idx (int): The index of the word for which to find the nearest neighbors.
            k (int, optional): The number of nearest neighbors to return. Defaults to 10.
        Returns:
            list[int]: A list of indices of the k nearest words.
        """
        word_emb = self.tag_emb(torch.Tensor(word_idx)).squeeze() # (emb_dim)
        distances = F.cosine_similarity(word_emb.unsqueeze(0), self.tag_emb.weight, dim=1)
        _, indices = distances.topk(k+1)
        return indices[1:].tolist()