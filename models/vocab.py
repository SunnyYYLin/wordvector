from functools import singledispatchmethod
from typing import Optional, Any, Sequence
import torch
from tqdm import tqdm
import json

UNK = '<UNK>'  # Special token for unknown words

class Vocabulary:
    """Vocabulary类用于构建和管理词汇表。
    属性:
        vocab (list[str]): 词汇表中的单词列表。
        word2idx (dict[str, int]): 从单词到索引的映射。
        freq (list[int]): 词汇表中每个单词的频率。
    方法:
        __init__(words: list[str], min_count: Optional[int] = None, max_vocab: Optional[int] = None) -> None:
            初始化Vocabulary对象，并根据给定的单词列表和阈值参数构建词汇表。
        _threshold_mode(min_count: Optional[int] = None, max_vocab: Optional[int] = None) -> None:
            设置词汇表的阈值模式，只能提供min_count或max_vocab中的一个。
        _build(words: list[str]) -> None:
            根据给定的单词列表和阈值参数构建词汇表。
        __len__() -> int:
            返回词汇表的大小。
        __getitem__(key: Any) -> Any:
            根据键的类型检索项目。
        frequency(word: str) -> int:
            返回词汇表中某个单词的频率。
        save(output_file: str) -> None:
            将词汇表保存到指定的文件中。
        load(path: str) -> 'Vocabulary':
            从指定的文件中加载词汇表。
    """
    def __init__(
        self,
        words: list[str],
        min_count: Optional[int] = None,
        max_vocab: Optional[int] = None,
    ) -> None:
        self._threshold_mode(min_count, max_vocab)

        self.vocab: list[str] = []          # List of words in the vocabulary
        self.word2idx: dict[str, int] = {}  # Mapping from word to index
        self.freq: list[int] = []           # Frequency of each word in the vocabulary

        if words:
            self._build(words)
        
    def _threshold_mode(self, min_count: Optional[int] = None, max_vocab: Optional[int] = None) -> None:
        if min_count is not None and max_vocab is not None:
            raise ValueError("Only one of min_count or max_vocab should be provided.")
        elif min_count is not None:
            self.threshold_mode = 'min_count'
            self.min_count = min_count
            self.max_vocab = None
        elif max_vocab is not None:
            self.threshold_mode = 'max_vocab'
            self.max_vocab = max_vocab
            self.min_count = None
        else:
            # Default mode
            self.threshold_mode = 'min_count'
            self.min_count = 1
            self.max_vocab = None

    def _build(self, words: list[str]) -> None:
        """Builds the vocabulary based on the tokenized sentences and thresholding parameters."""
        # Count word frequencies
        word2freq: dict[str, int] = {}
        
        for word in tqdm(words, desc='Traversing words'):
            word2freq[word] = word2freq.get(word, 0) + 1
        
        print(f"Total unique words: {len(word2freq)}")
        print("Sorting words based on frequency...")
        word_freq = sorted(word2freq.items(), key=lambda x: x[1], reverse=True)
        del(word2freq)

        # Filter words based on threshold mode
        print(f"Filtering words based on {self.threshold_mode}...")
        match self.threshold_mode:
            case 'min_count':
                self.vocab = [word for word, freq in word_freq if freq >= self.min_count]
                unk_count = sum(freq for word, freq in word_freq if freq < self.min_count)
            case 'max_vocab':
                self.vocab = [word for word, freq in word_freq[:self.max_vocab-1]]
                unk_count = sum(freq for word, freq in word_freq[self.max_vocab-1:])
            case _:
                raise ValueError(f"Unsupported threshold mode: {self.threshold_mode}")

        self.vocab.append(UNK)
        print("Building word2idx mapping...")
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        print("Building frequency list...")
        self.freq = [freq for word, freq in word_freq].append(unk_count)
        del(word_freq)

    def __len__(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    @singledispatchmethod
    def __getitem__(self, key: Any) -> Any:
        """Retrieves item based on the type of key."""
        raise NotImplementedError(f'Unsupported key type: {type(key)}')

    @__getitem__.register
    def _(self, key: int) -> str:
        """Gets the word corresponding to the given index."""
        return self.vocab[key]

    @__getitem__.register
    def _(self, key: str) -> int:
        """Gets the index corresponding to the given word."""
        return self.word2idx.get(key, self.word2idx[UNK])
    
    @__getitem__.register
    def _(self, key: list) -> list:
        return [self[word] for word in key]
    
    @__getitem__.register
    def _(self, key: tuple) -> tuple:
        return tuple(self[word] for word in key)

    def __contains__(self, word: str) -> bool:
        """Checks if a word is in the vocabulary."""
        return word in self.word2idx

    def frequency(self, word: str) -> int:
        """Returns the frequency of a word in the vocabulary."""
        return self.freq[self.word2idx[word]]
    
    def save(self, output_file: str) -> None:
        torch.save({
            'vocab': self.vocab,
            'word2idx': self.word2idx,
            'freq': self.freq,
            'threshold_mode': self.threshold_mode,
            'min_count': getattr(self, 'min_count', None),
            'max_vocab': getattr(self, 'max_vocab', None)
        }, output_file)
            
    @staticmethod
    def load(path: str) -> 'Vocabulary':
        checkpoint = torch.load(path)
        vocab = Vocabulary([])
        vocab.vocab = checkpoint['vocab']
        vocab.word2idx = checkpoint['word2idx']
        vocab.freq = checkpoint['freq']
        vocab.threshold_mode = checkpoint['threshold_mode']
        vocab.min_count = checkpoint['min_count']
        vocab.max_vocab = checkpoint['max_vocab']
        return vocab