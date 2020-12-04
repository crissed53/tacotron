from collections import namedtuple
from dataclasses import dataclass
import os
from typing import List, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from . import (WAV_SAVE_DIR, MEL_SPEC_SAVE_DIR,
               METADATA_FILE, LIN_SPEC_SAVE_DIR)
from data.text import load_transcription, tokenize_transcription


@dataclass
class TorchLJSpeechBatch:
    uid_idx: torch.LongTensor
    lin_spec: torch.Tensor
    mel_spec: torch.Tensor
    emb_idx: torch.LongTensor
    num_sequences: torch.LongTensor

    def to(self, device: torch.device):
        return TorchLJSpeechBatch(
            self.uid_idx.to(device),
            self.lin_spec.to(device),
            self.mel_spec.to(device),
            self.emb_idx.to(device),
            self.num_sequences.to(device))


@dataclass
class TorchLJSpeechData:
    uid_idx: int
    lin_spec: torch.Tensor
    mel_spec: torch.Tensor
    emb_idx: torch.LongTensor

    def add_batch_dim(self) -> TorchLJSpeechBatch:
        return TorchLJSpeechBatch(
            uid_idx=torch.LongTensor([self.uid_idx]),
            lin_spec=self.lin_spec.unsqueeze(0),
            mel_spec=self.mel_spec.unsqueeze(0),
            emb_idx=self.emb_idx.unsqueeze(0),
            num_sequences=torch.LongTensor([self.emb_idx.size(0)])
        )


class TorchLJSpeechDataset(Dataset):
    tokenizer = tokenize_transcription(METADATA_FILE)
    spec_filename_format = '{}.npy'

    def __init__(self, num_data: int = None):
        """
        Initialize dataset for LJ Speech Dataset
        Args:
            num_data: number of data points to load. If None, get all the
                possible data
        """
        self.wav_save_dir = WAV_SAVE_DIR
        self.mel_spec_dir = MEL_SPEC_SAVE_DIR
        self.lin_spec_dir = LIN_SPEC_SAVE_DIR
        self.metadata_file = METADATA_FILE

        self.uid_to_transcription = load_transcription(self.metadata_file)

        if num_data is not None:
            reduced_items = list(self.uid_to_transcription.items())[:num_data]
            self.uid_to_transcription = {key: val for key, val in reduced_items}

        self.uids = list(self.uid_to_transcription.keys())

    def __getitem__(self, idx) -> TorchLJSpeechData:
        uid = self.uids[idx]

        # Transpose spectrograms so that dimension is in the form of
        # (seq, freq), which is the expected input shape for Tacotron

        lin_spec = torch.from_numpy(
            np.load(
                os.path.join(
                    self.lin_spec_dir, self.spec_filename_format.format(uid)
                )
            ).T
        )

        mel_spec = torch.from_numpy(
            np.load(
                os.path.join(
                    self.mel_spec_dir, self.spec_filename_format.format(uid)
                )
            ).T
        )

        transcription = self.uid_to_transcription[uid]

        emb_idx = torch.LongTensor(
            self.tokenizer.get_idx_from_string(transcription))

        return TorchLJSpeechData(idx, lin_spec, mel_spec, emb_idx)

    def __len__(self):
        return len(self.uids)

    @classmethod
    def batch_tacotron_input(
            cls, batch: List[TorchLJSpeechData]) -> TorchLJSpeechBatch:
        """
        Collate a given list of data into batches
        Args:
            batch: list of Tacotron input data

        Returns:
            batched input data for Tacotron

        """
        input_lengths = torch.LongTensor([item.emb_idx.size(0)
                                          for item in batch])
        input_lengths, sorted_idx = input_lengths.sort(descending=True)

        # Permute batch according to input sequence lengths
        batch = [batch[i] for i in sorted_idx]

        uid_idx = torch.LongTensor([item.uid_idx for item in batch])

        # Stack spectrograms along batch axis; pad zeros for any unbalanced
        # number of frames
        lin_spec = pad_sequence(
            [item.lin_spec for item in batch],
            batch_first=True
        )

        mel_spec = pad_sequence(
            [item.mel_spec for item in batch],
            batch_first=True
        )

        # Pad embedding indices if necessary
        emb_idx = pad_sequence(
            [item.emb_idx for item in batch],
            batch_first=True,
            padding_value=cls.tokenizer.pad_idx
        )

        return TorchLJSpeechBatch(uid_idx=uid_idx,
                                  lin_spec=lin_spec,
                                  mel_spec=mel_spec,
                                  emb_idx=emb_idx,
                                  num_sequences=input_lengths)
