import random
import json
import math
from itertools import product
import torch
from tqdm import tqdm
from .vocab import Vocabulary

START = '<START>'
END = '<END>'
UNK = '<UNK>'

class CBOWPairs:
    """CBOWPairs 类用于存储和处理连续词袋模型（CBOW）中的训练数据对。
    参数:
        bags (tuple[tuple[int]]): 词袋的索引元组。
        tags (tuple[int]): 中心词的索引元组。
        negatives (tuple[tuple[int]]): 负采样词的索引元组。
    属性:
        bags (torch.Tensor): 转换为长整型的词袋张量。
        tags (torch.Tensor): 转换为长整型的标签张量。
        negatives (torch.Tensor): 转换为长整型的负采样词张量。
    方法:
        __len__() -> int: 返回目标词的数量。
        __str__() -> str: 返回 CBOWPairs 对象的字符串表示。
        to(device) -> CBOWPairs: 将所有张量移动到指定设备，并返回自身。
    """
    def __init__(self, 
                 bags: tuple[tuple[int]], 
                 tags: tuple[int], 
                 negatives: tuple[tuple[int]]) -> None:
        self.bags = torch.Tensor(bags).long()
        self.tags = torch.Tensor(tags).long()
        self.negatives = torch.Tensor(negatives).long()
        assert len(bags) == len(tags) == len(negatives), \
            'Batch size of bags, tags and negatives should be the same'
    
    def __len__(self) -> int:
        return len(self.tags)
    
    def __str__(self) -> str:
        return f"""CBOWPairs(
    bags: {self.bags},
    tags: {self.tags},
    negatives: {self.negatives}
)"""

    def to(self, device):
        self.bags = self.bags.to(device)
        self.tags = self.tags.to(device)
        self.negatives = self.negatives.to(device)
        return self

class CBOWDataSet:
    def __init__(self, input_file: str, 
                 window_size: int|None=None, 
                 min_count: int|None=None, 
                 max_vocab: int|None=None) -> None:
        self.input_file = input_file
        self.window_size = window_size
        
        # data
        self.sentences: list[list[int]] = []  # [[idx1, idx2, ...], ...]
        self.coords: list[tuple[int]] = []  # [(idx_s, idx_w), ...]
        
        # tokenized.txt
        if self.input_file.endswith('.txt'):
            self._process_data()
            self.vocab = Vocabulary(self.sentences2words(self.sentences), min_count, max_vocab)
            self.vocab_size = len(self.vocab)
            self._to_indices()
        # dataset.json
        elif self.input_file.endswith('.json'):
            self._load_data()
        else:
            raise ValueError('Unsupported file format')
        
        self.generate_coords()
    
    def __len__(self) -> int:
        return len(self.coords)
    
    def _process_data(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Processing data'):
                line = line.strip().split(' ')
                self.sentences.append(line)
        
        self._padding_sentence()
        
    def save(self, output_file: str):
        with open(output_file, 'w') as f:
            json.dump({
                'window_size': self.window_size,
                'sentences': self.sentences,
                'vocab_size': self.vocab_size
            }, f)
            
    def _load_data(self):
        with open(self.input_file, 'r') as f:
            data = json.load(f)
            self.window_size = data['window_size']
            self.sentences = data['sentences']
            self.vocab_size = data['vocab_size']
            
    def _padding_sentence(self):
        head = [START] * self.window_size
        tail = [END] * self.window_size
        self.sentences = [head + sentence + tail for sentence in self.sentences]
        
    def _to_indices(self):
        self.sentences = [self.vocab[sentence] for sentence in tqdm(self.sentences, desc='Converting sentences to indices')]
        
    @staticmethod
    def sentences2words(sentences: list[list[str]]) -> list[str]:
        return [word for sentence in sentences for word in sentence]
        
    def generate_coords(self):
        for idx_s, sentence in enumerate(tqdm(self.sentences, desc='Generating coordinates')):
            for idx_w in range(self.window_size, len(sentence)-self.window_size):
                self.coords.append((idx_s, idx_w))
                
    def partition(self,
                batch_size:int,
                ratio:float=0.9, 
                shuffle:bool=True,
                neg_size:int=5) -> tuple['CBOWDataLoader']:
        total = len(self)
        train_size = math.ceil(total * ratio)
        
        coords = self.coords.copy()
        if shuffle:
            print('Shuffling coordinates...')
            random.shuffle(coords)
        print('Partitioning dataset...')
        train_coords = coords[:train_size]
        test_coords = coords[train_size:]
        return (
            CBOWDataLoader(self, train_coords, batch_size, neg_size),
            CBOWDataLoader(self, test_coords, batch_size, neg_size)
        )
        
class CBOWDataLoader:
    """
    CBOWDataLoader 类用于加载连续词袋（CBOW）模型的数据。
    参数:
        dataset (CBOWDataSet): 包含训练数据的 CBOW 数据集。
        coords (list[tuple[int]]): 数据集中的坐标列表，每个坐标表示一个目标词的位置。
        batch_size (int): 每个批次的大小。
        neg_size (int): 每个目标词的负采样数量。
    方法:
        __len__() -> int:
            返回数据集的批次数量。
        __iter__():
            迭代器方法，生成包含目标词、上下文词袋和负采样词的批次。
        _get_negatives(tag) -> tuple[int]:
            根据目标词生成负采样词。
    """
    def __init__(self,
                 dataset: CBOWDataSet,
                 coords: list[tuple[int]],
                 batch_size: int,
                 neg_size: int) -> None:
        self.dataset = dataset
        self.coords = coords
        self.batch_size = batch_size
        self.neg_size = neg_size
        
    def __len__(self) -> int:
        return len(self.coords) // self.batch_size

    def __iter__(self):
        tags, bags, negatives = [], [], []
        padding = self.dataset.window_size
        for coord in self.coords:
            tag = self.dataset.sentences[coord[0]][coord[1]]
            bag = self.dataset.sentences[coord[0]][coord[1]-padding:coord[1]] + \
                  self.dataset.sentences[coord[0]][coord[1]+1:coord[1]+padding+1]
            negative = self._get_negatives(tag)
            tags.append(tag)
            bags.append(tuple(bag))
            negatives.append(negative)
            if len(tags) == self.batch_size:
                yield CBOWPairs(bags, tags, negatives)
                tags, bags, negatives = [], [], []
            
    def _get_negatives(self, tag) -> tuple[int]:
        negatives = []
        while len(negatives) < self.neg_size:
            negative = random.randint(0, self.dataset.vocab_size-1)
            if negative != tag:
                negatives.append(negative)
        return tuple(negatives)
    
    def partition(self, ratio: float, shuffle: bool=True) -> tuple["CBOWDataLoader"]:
        total = len(self.coords)
        train_size = math.ceil(total * ratio)
        
        coords = self.coords.copy()
        if shuffle:
            print('Shuffling coordinates...')
            random.shuffle(coords)
        train_coords = coords[:train_size]
        test_coords = coords[train_size:]
        return (
            CBOWDataLoader(self.dataset, train_coords, self.batch_size, self.neg_size),
            CBOWDataLoader(self.dataset, test_coords, self.batch_size, self.neg_size)
        )
        
        
if __name__ == '__main__':
    dataset = CBOWDataSet('data/cn/dataset.json')
    train_loader, valid_loader, test_loader = dataset.partition(8, neg_size=3)
    for pairs in train_loader:
        print(pairs)
        break
    print(len(train_loader), len(valid_loader), len(test_loader))