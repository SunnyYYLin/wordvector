import random
import json
import numpy as np
from tqdm import tqdm

START = '<START>'
END = '<END>'

class CBOWPairs:
    def __init__(self, 
                 bags: tuple[tuple[int]], 
                 targets: tuple[int], 
                 negatives: tuple[tuple[int]]) -> None:
        self.bags = bags
        self.targets = targets
        self.negatives = negatives
        assert len(bags) == len(targets) == len(negatives), 'Batch size of bags, targets and negatives should be the same'
        
    @property
    def batch_size(self) -> int:
        return len(self.bags)

class CBOWDataSet:
    def __init__(self, input_file: str, min_count: int=0, window_size: int=5) -> None:
        self.input_file = input_file
        self.min_count = min_count
        self.window_size = window_size
        self.word_freq: dict[str, int] = {}  # 修改为字典形式
        self.sentences: list[list[str]] = []  # 存储句子列表
        if self.input_file.endswith('.txt'):
            self._process_data()
        elif self.input_file.endswith('.json'):
            self._load_data()
        self._padding_sentence()

    def __iter__(self):
        random.shuffle(self.sentences)
        return self

    def __next__(self) -> CBOWPairs:
        pass

    def _init_vocab(self):
        self.index_to_word = list(self.word_count.keys())

    def _process_data(self):
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip().split(' ')
                self.sentences.append(line)
                for word in line:
                    if word not in self.word_count:
                        self.word_count[word] = 0
                    self.word_count[word] += 1
        self.data = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        self._initialize_vocab()

    def _padding_sentence(self):
        padding = self.window_size // 2
        head = [START] * padding
        tail = [END] * padding
        self.sentences = [head + sentence + tail for sentence in self.sentences]

    def _load_data(self):
        with open(self.input_file, 'r') as f:
            data = json.load(f)
            self.sentences = data['sentences']
            self.word_count = data['word_count']
            self.data = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        self._initialize_vocab()