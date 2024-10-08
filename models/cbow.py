import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    """CBOW 类实现了连续词袋模型，用于词向量的训练和预测。
    方法:
        __init__(self, vocab_size, emb_dim, shared_embeddings=False):
            初始化 CBOW 模型。
            参数:
                vocab_size (int): 词汇表的大小。
                emb_dim (int): 词向量的维度。
                shared_embeddings (bool, 可选): 是否共享输入和输出的词向量。默认为 False。
        average(self, bags: torch.Tensor) -> torch.Tensor:
            计算每个词袋的平均词向量。
            参数:
                bags (torch.Tensor): 形状为 (batch_size, 2*window_size) 的张量，包含每个词袋中的词索引。
            返回:
                torch.Tensor: 形状为 (batch_size, emb_dim) 的张量，包含每个词袋的平均词向量。
        forward(self, bags: torch.Tensor) -> torch.Tensor:
            执行 CBOW 模型的前向传播。
            参数:
                bags (torch.Tensor): 形状为 (batch_size, 2*window_size) 的张量，包含每个词袋中的词索引。
            返回:
                torch.Tensor: 形状为 (batch_size, vocab_size) 的张量，包含词汇表的概率分布。
        predict(self, bags: torch.Tensor) -> torch.Tensor:
            预测给定词袋的类别标签。
            参数:
                bags (torch.Tensor): 形状为 (batch_size, 2*window_size) 的张量，包含每个词袋中的词索引。
            返回:
                torch.Tensor: 形状为 (batch_size,) 的张量，包含预测的类别标签。
        loss(self, pair, use_logsigmoid=True) -> torch.Tensor:
            计算给定词袋、标签和负样本的损失。
            参数:
                pair (object): 包含以下属性的对象:
                    - bags (torch.Tensor): 形状为 (batch_size, 2*window_size, emb_dim) 的张量，表示上下文词向量。
                    - tags (torch.Tensor): 形状为 (batch_size, emb_dim) 的张量，表示标签词向量。
                    - negatives (torch.Tensor): 形状为 (batch_size, neg_size, emb_dim) 的张量，表示负样本词向量。
                use_logsigmoid (bool, 可选): 是否使用对数 Sigmoid 函数计算损失。默认为 True。
            返回:
                torch.Tensor: 表示平均损失的标量张量。
        _pos_loss(self, avg: torch.Tensor, tags: torch.Tensor, use_logsigmoid=True) -> torch.Tensor:
            计算正样本的损失。
            参数:
                avg (torch.Tensor): 形状为 (batch_size, emb_dim) 的张量，表示平均词向量。
                tags (torch.Tensor): 形状为 (batch_size, emb_dim) 的张量，表示标签词向量。
                use_logsigmoid (bool, 可选): 是否使用对数 Sigmoid 函数计算损失。默认为 True。
            返回:
                torch.Tensor: 形状为 (batch_size) 的张量，表示正样本的损失。
        _neg_loss(self, avg: torch.Tensor, negatives: torch.Tensor, use_logsigmoid=True) -> torch.Tensor:
            计算负样本的损失。
            参数:
                avg (torch.Tensor): 形状为 (batch_size, emb_dim) 的张量，表示平均词向量。
                negatives (torch.Tensor): 形状为 (batch_size, num_negatives, emb_dim) 的张量，表示负样本词向量。
                use_logsigmoid (bool, 可选): 是否使用对数 Sigmoid 函数计算损失。默认为 True。
            返回:
                torch.Tensor: 形状为 (batch_size) 的张量，表示负样本的损失。
        nearest(self, word_idx: int, k=10) -> list[int]:
            基于余弦相似度找到给定词索引的最近邻词。
            参数:
                word_idx (int): 要查找最近邻的词索引。
                k (int, 可选): 要返回的最近邻词的数量。默认为 10。
            返回:
                list[int]: 包含 k 个最近邻词索引的列表。
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            neg_loss = neg_loss.exp().sum(dim=1).log()
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
        word_emb = self.tag_emb(torch.tensor([word_idx], dtype=torch.long).to(self.device)).squeeze() # (emb_dim)
        distances = F.cosine_similarity(word_emb.unsqueeze(0), self.tag_emb.weight, dim=1)
        _, indices = distances.topk(k+1)
        return indices[1:].tolist()
    
    def save(self, path: str) -> None:
        """
        Save the model to the given path.
        Args:
            path (str): The path to save the model.
        """
        checkpoint = {
            'vocab_size': self.tag_emb.num_embeddings,
            'emb_dim': self.tag_emb.embedding_dim,
            'shared_embeddings': self.bag_emb.weight is self.tag_emb.weight,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, path)
        
    @classmethod
    def load(cls, path: str) -> 'CBOW':
        """
        Load the model from the given path.
        Args:
            path (str): The path to load the model from.
        Returns:
            CBOW: The loaded CBOW model.
        """
        checkpoint = torch.load(path)
        model = cls(checkpoint['vocab_size'], checkpoint['emb_dim'], checkpoint['shared_embeddings'])
        model.load_state_dict(checkpoint['state_dict'])
        return model