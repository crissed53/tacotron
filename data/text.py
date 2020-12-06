from dataclasses import dataclass
from string import punctuation, ascii_lowercase
from typing import Union, Dict, List

import torch
import torch.nn as nn

from . import METADATA_FILE


@dataclass
class TextData:
    uid: str
    transcription: str


class CharTokenizer:
    def __init__(self, init_with_basic_letters: bool = True):
        self.ctoi = {'<pad>': 0, '<unk>': 1}
        self.itoc = {0: '<pad>', 1: '<unk>'}

        self.pad_idx = 0
        self.unk_idx = 1

        self.cnt = 2

        if init_with_basic_letters:
            self.add_from_string(punctuation)
            self.add_from_string(ascii_lowercase)

    def __len__(self):
        return self.cnt

    def add_from_string(self, string: Union[str, list]) -> None:
        for char in set(string):
            if char not in self.ctoi:
                self.ctoi[char] = self.cnt
                self.itoc[self.cnt] = char
                self.cnt += 1

    def get_idx_from_string(self, string: str) -> List[int]:
        return [self.ctoi.get(char, self.unk_idx) for char in string]


class TextModel:
    """Model for exchanging text input into embedding for Tacotron"""
    def __init__(self, embedding_dim: int = 256):
        self.tokenizer = tokenize_transcription(METADATA_FILE)
        self.embedding = nn.Embedding(num_embeddings=len(self.tokenizer),
                                      embedding_dim=embedding_dim,
                                      padding_idx=self.tokenizer.pad_idx)

    def embedding_from_text(self, text: str,
                            expand_dim: bool = False) -> torch.FloatTensor:
        """
        Create character embeddings from a text
        Args:
            text: text to convert to the embeddings
            expand_dim: if set True, expand dimension of the resulting tensor
                along the batch dimension

        Returns:

        """
        idx = torch.LongTensor(self.tokenizer.get_idx_from_string(text))
        # expand batch dim
        embedding = self.embedding(idx)
        if expand_dim:
            embedding = embedding.unsqueeze(0)
        return embedding


def decode_single_line(line: str, normalized: bool = True) -> TextData:
    """Decode a single line of metadata.csv"""
    uid, trs, n_trs = line.strip().split('|')
    transcription = n_trs if normalized else trs
    return TextData(uid, transcription.lower())


def tokenize_transcription(
        metadata_file: str, normalized: bool = True) -> CharTokenizer:
    """Tokenize all the transcriptions in characters"""
    tokenizer = CharTokenizer(init_with_basic_letters=False)

    with open(metadata_file) as meta_file:
        for line in meta_file:
            txt_data = decode_single_line(line, normalized=normalized)
            tokenizer.add_from_string(txt_data.transcription)

    return tokenizer


def load_transcription(metadata_file: str,
                       normalized: bool = True) -> Dict[str, str]:
    uid_to_transcription = {}

    with open(metadata_file) as meta_file:
        for line in meta_file:
            txt_data = decode_single_line(line, normalized=normalized)
            uid_to_transcription[txt_data.uid] = txt_data.transcription

    return uid_to_transcription
