import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
from .cbow import CBOW
from .dataset import CBOWDataLoader
from .vocab import Vocabulary

class Word2Vec:
    """Word2Vec 类用于训练和使用词向量模型。
    方法:
        __init__(model: CBOW, vocab: Vocabulary) -> None:
            初始化 Word2Vec 类。
        train(loader: CBOWDataLoader, epochs=10, lr=0.001, device='cuda', train_ratio=0.9, log_dir='./logs'):
            训练词向量模型。
        test(loader: CBOWDataLoader, device='cuda' if torch.cuda.is_available() else 'cpu'):
            测试词向量模型。
        nearest(word: str) -> list[str]:
            查找与给定词最相近的词。
    """
    def __init__(self, model: CBOW, vocab: Vocabulary) -> None:
        self.model = model
        self.vocab = vocab

    def train(self, 
            loader: CBOWDataLoader,
            epochs=10, 
            lr=0.001,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            train_ratio=0.9,
            log_dir='./logs',
            use_logsigmoid=True):

        writer = SummaryWriter(log_dir)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.to(device)

        for epoch in range(epochs):
            train_loader, valid_loader = loader.partition(train_ratio)
            self.model.train()
            
            for batch_idx, pair in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
                pair = pair.to(device)
                optimizer.zero_grad()

                loss = self.model.loss(pair, use_logsigmoid=use_logsigmoid)
                loss.backward()
                optimizer.step()

                # 记录损失
                writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)

                # 每 100 个批次计算并记录一次训练准确率
                if (batch_idx + 1) % 100 == 0:
                    output = self.model.predict(pair.bags)
                    accuracy = torch.sum(output == pair.tags).item() / len(pair.tags)
                    writer.add_scalar('Training Accuracy', accuracy, epoch * len(train_loader) + batch_idx)

            # 模型验证
            self.model.eval()
            total_val_loss = 0
            total_val_accuracy = 0
            for pair in tqdm(valid_loader, desc='Validation'):
                pair = pair.to(device)
                val_loss = self.model.loss(pair)
                total_val_loss += val_loss.item()

                # 计算并记录验证准确率
                output = self.model.predict(pair.bags)
                val_accuracy = torch.sum(output == pair.tags).item() / len(pair.tags)
                total_val_accuracy += val_accuracy

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_accuracy = total_val_accuracy / len(valid_loader)
            print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}')

            # 使用 TensorBoard 记录验证集的损失和准确率
            writer.add_scalar('Validation Loss', avg_val_loss, epoch)
            writer.add_scalar('Validation Accuracy', avg_val_accuracy, epoch)
            
        writer.close()
        
    def test(self, 
             loader: CBOWDataLoader, 
             device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model.eval()
        total_accuracy = 0
        for pair in tqdm(loader, desc='Testing'):
            pair = pair.to(device)
            output = self.model.predict(pair.bags)
            accuracy = torch.sum(output == pair.tags).item() / len(pair.tags)
            total_accuracy += accuracy
        avg_accuracy = total_accuracy / len(loader)
        print(f'Test Accuracy: {avg_accuracy:.4f}')
        return avg_accuracy
    
    @staticmethod
    def load(path: str, 
             device='cuda' if torch.cuda.is_available() else 'cpu') -> 'Word2Vec':
        vocab = Vocabulary.load(os.path.join(path, 'vocab.pth'))
        model = CBOW.load(os.path.join(path, 'model.pth'))
        model.to(device)
        return Word2Vec(model, vocab)
    
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'model.pth'))
        self.vocab.save(os.path.join(path, 'vocab.pth'))
        
    def nearest(self, word: str) -> list[str]:
        idx = self.vocab[word]
        neighbor_idx = self.model.nearest(idx)
        return self.vocab[neighbor_idx]
    
    def sim(self, word1: str, word2: str) -> float:
        idx1, idx2 = self.vocab[word1], self.vocab[word2]
        sim = self.model.bag_emb.weight[idx1] @ self.model.tag_emb.weight[idx2]