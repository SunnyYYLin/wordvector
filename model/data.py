import random
import json
import math
from itertools import product
import torch
from tqdm import tqdm

START = '<START>'
END = '<END>'
UNK = '<UNK>'

class CBOWPairs:
    def __init__(self, 
                 bags: tuple[tuple[int]], 
                 targets: tuple[int], 
                 negatives: tuple[tuple[int]]) -> None:
        self.bags = torch.Tensor(bags).long()
        self.targets = torch.Tensor(targets).long()
        self.negatives = torch.Tensor(negatives).long()
        assert len(bags) == len(targets) == len(negatives), \
            'Batch size of bags, targets and negatives should be the same'
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __str__(self) -> str:
        return f"""CBOWPairs(
    bags: {self.bags},
    targets: {self.targets},
    negatives: {self.negatives}
)"""

    def to(self, device):
        self.bags = self.bags.to(device)
        self.targets = self.targets.to(device)
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
        self.vocab: list[str] = []  # [word1, word2, ...]
        self.freq: list[int] = []  # [freq1, freq2, ...]
        self.word2idx: dict[str, int] = {} # {word: idx, ...}
        self.coords: list[tuple[int]] = []  # [(idx_s, idx_w), ...]
        
        # threshold_mode: 'min_count' or 'max_vocab'
        if min_count:
            self.min_count = min_count
            self.threshold_mode = 'min_count'
        elif max_vocab:
            self.max_vocab = max_vocab
            self.threshold_mode = 'max_vocab'
        else:
            self.threshold_mode = None
        
        # tokenized.txt
        if self.input_file.endswith('.txt'):
            self._process_data()
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
        self._init_vocab()
        
    def save(self, output_file: str):
        with open(output_file, 'w') as f:
            json.dump({
                'window_size': self.window_size,
                'vocab': self.vocab,
                'freq': self.freq,
                'word2idx': self.word2idx,
                'sentences': self.sentences
            }, f)
            
    def _load_data(self):
        with open(self.input_file, 'r') as f:
            data = json.load(f)
            self.sentences = data['sentences']
            self.vocab = data['vocab']
            self.freq = data['freq']
            self.word2idx = data['word2idx']
            self.window_size = data['window_size']
            
    def _padding_sentence(self):
        padding = self.window_size // 2
        head = [START] * padding
        tail = [END] * padding
        self.sentences = [head + sentence + tail for sentence in self.sentences]
            
    def _init_vocab(self):
        words = [word for sentence in self.sentences for word in sentence]
        word2freq = {}
        for word in tqdm(words, desc='Counting word frequency'):
            word2freq[word] = word2freq.get(word, 0) + 1
        print('Sorting word frequency...')
        word_freq = sorted(word2freq.items(), key=lambda x: x[1], reverse=True)
        
        print('Removing low frequency words...')
        match self.threshold_mode:
            case 'min_count':
                low_freq_count = sum([freq for word, freq in word_freq if freq < self.min_count])
                word_freq = [(word, freq) for word, freq in word_freq if freq >= self.min_count]
                word_freq.append((UNK, low_freq_count))
            case 'max_vocab':
                low_freq_count = sum([freq for word, freq in word_freq[self.max_vocab-1:]])
                word_freq = word_freq[:self.max_vocab-1]
                word_freq.append((UNK, low_freq_count))
            case _:
                pass
        
        print('Building vocabulary...')
        self.vocab = [word for word, freq in word_freq]
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.freq = [freq for word, freq in word_freq]
        
        idx_sentences = []
        for sentence in tqdm(self.sentences, desc='Converting sentences to indices'):
            idx_sentences.append([self.word2idx.get(word, self.word2idx[UNK]) for word in sentence])
        self.sentences = idx_sentences
        
    def generate_coords(self):
        for idx_s, sentence in enumerate(tqdm(self.sentences, desc='Generating coordinates')):
            for idx_w in range(self.window_size//2, len(sentence)-self.window_size//2):
                self.coords.append((idx_s, idx_w))
                
    def partition(self,
                batch_size:int,
                ratio:tuple[float]=(0.8, 0.1, 0.1), 
                shuffle:bool=True,
                neg_size:int=5) -> tuple['CBOWDataLoader']:
        assert math.isclose(sum(ratio), 1), 'Sum of ratio should be 1'
        assert len(ratio) == 3, 'Length of ratio should be 3'
        total = len(self)
        train_size = int(total * ratio[0])
        valid_size = int(total * ratio[1])
        test_size = total - train_size - valid_size
        
        coords = self.coords.copy()
        if shuffle:
            print('Shuffling coordinates...')
            random.shuffle(coords)
        print('Partitioning dataset...')
        train_coords = coords[:train_size]
        valid_coords = coords[train_size:train_size+valid_size]
        test_coords = coords[train_size+valid_size:]
        return (
            CBOWDataLoader(self, train_coords, batch_size, neg_size),
            CBOWDataLoader(self, valid_coords, batch_size, neg_size),
            CBOWDataLoader(self, test_coords, batch_size, neg_size)
        )
        
class CBOWDataLoader:
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
        targets, bags, negatives = [], [], []
        padding = self.dataset.window_size // 2
        for coord in self.coords:
            target = self.dataset.sentences[coord[0]][coord[1]]
            bag = self.dataset.sentences[coord[0]][coord[1]-padding:coord[1]] + \
                  self.dataset.sentences[coord[0]][coord[1]+1:coord[1]+padding+1]
            negative = self._get_negatives(target)
            targets.append(target)
            bags.append(tuple(bag))
            negatives.append(negative)
            if len(targets) == self.batch_size:
                yield CBOWPairs(bags, targets, negatives)
                targets, bags, negatives = [], [], []
            
    def _get_negatives(self, target) -> tuple[int]:
        negatives = []
        while len(negatives) < self.neg_size:
            negative = random.randint(0, len(self.dataset.vocab)-1)
            if negative != target:
                negatives.append(negative)
        return tuple(negatives)
        
if __name__ == '__main__':
    dataset = CBOWDataSet('data/cn/dataset.json')
    train_loader, valid_loader, test_loader = dataset.partition(8, neg_size=3)
    for pairs in train_loader:
        print(pairs)
        break
    print(len(train_loader), len(valid_loader), len(test_loader))