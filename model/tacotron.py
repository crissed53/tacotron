from collections import namedtuple
from dataclasses import dataclass
import os
from typing import Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .module import Cbhg, PreNet, DecoderRnn, ContentBasedAttention, SeqModule
from data.text import TextModel
from data.audio import AudioProcessParam, AudioProcessingHelper
from data.torch import TorchLJSpeechBatch


@dataclass
class DecoderOutput:
    pred_mel_spec: torch.Tensor
    attention_weight: torch.Tensor


@dataclass
class TacotronOutput:
    pred_lin_spec: torch.Tensor
    pred_mel_spec: torch.Tensor
    attention_weight: torch.Tensor


class Tacotron:
    """The architecture of Tacotron consisting of text model, neural network,
    and reconstruction module from predicted linear spectrogram"""
    def __init__(self, r: int = 2, embedding_dim: int = 256):
        self.nn = TacotronModel(r, embedding_dim)
        self.text_model = TextModel(embedding_dim=embedding_dim)

    def to(self, device: torch.device) -> None:
        self.nn = self.nn.to(device)
        self.text_model.embedding = self.text_model.embedding.to(device)

    def save(self, save_file: str, device: torch.device) -> None:
        torch.save(
            {'model_state_dict': self.nn.cpu().state_dict(),
             'embedding': self.text_model.embedding.cpu().state_dict()
             }, save_file
        )

        self.nn.to(device)
        self.text_model.embedding.to(device)

    def load(self, save_file: str,
             device: torch.device = torch.device('cpu')) -> None:
        checkpoint = torch.load(save_file, map_location=device)
        self.nn.load_state_dict(checkpoint['model_state_dict'])
        self.text_model.embedding.load_state_dict(checkpoint['embedding'])

    def text2taco_output(self, text: str) -> TacotronOutput:
        """Converts a text to Tacotron Output"""
        text = text.lower()
        char_embedding = self.text_model.embedding_from_text(
            text, expand_dim=True)
        return self.forward_inference(char_embedding)

    def text2audio(self, text: str) -> np.ndarray:
        output = self.text2taco_output(text)
        lin_spec = output.pred_lin_spec.cpu().numpy()
        return AudioProcessingHelper.spec2audio(lin_spec)

    def forward_train(self, batch_data: TorchLJSpeechBatch) -> TacotronOutput:
        embedding = self.text_model.embedding(batch_data.emb_idx)
        return self.nn.forward(embedding, batch_data.mel_spec)

    def forward_inference(self, x: torch.Tensor,
                          low_amp_thres: float = -10,
                          stop_low_energy_count: int = 40,
                          max_frame_len: int = 180) -> TacotronOutput:
        """
        Forward pass for Tacotron at inference stage
        Args:
            x: input character embedding. Dim: (batch, seq, feature)
            low_amp_thres: average energy level of r generated frames lower
                than this is considered silent
            stop_low_energy_count: if silent frames are generated continuously
                upto this number, stop the inference stage immediately
            max_frame_len: maximum number of the generated spectrogram

        Returns:
            predicted linear and mel spectrogram, and attention weight

        """
        with torch.no_grad():
            encoder_states = self.nn.encoder(x)

            go_frame = self.nn.get_go_frame(x.shape[0]).to(x.device)
            last_frame = go_frame

            low_amp_count = 0
            pred_mel_spec = []
            attention_weights = []
            frame_count = 0
            while (low_amp_count < stop_low_energy_count
                   and frame_count < max_frame_len):
                decoder_output = self.nn.decoder.forward(
                    last_frame, encoder_states)
                pred_mel_frames = decoder_output.pred_mel_spec
                pred_mel_spec.append(pred_mel_frames)

                attention_weights.append(decoder_output.attention_weight)

                last_frame = pred_mel_frames[:, -1].unsqueeze(1)

                frame_count += pred_mel_frames.size()[1]

                avg_amp = torch.mean(pred_mel_frames).item()
                if avg_amp < low_amp_thres:
                    # add r
                    low_amp_count += pred_mel_frames.size()[1]
                else:
                    low_amp_count = 0

            pred_mel_spec = torch.cat(pred_mel_spec, dim=1)
            if frame_count < max_frame_len:
                # concatenate the predicted frames while
                # discarding low energy content
                pred_mel_spec = pred_mel_spec[:, :-low_amp_count]

            # concatenate the predicted frames along decoder seq axis
            attention_weight = torch.cat(attention_weights, dim=1)

            pred_lin_spec = self.nn.fc_lin_spec_target(
                self.nn.cbhg(pred_mel_spec))

            return TacotronOutput(
                pred_lin_spec, pred_mel_spec, attention_weight)


class TacotronModel(nn.Module):
    """Module encapsulating language model, encoder, attention module
    and decoder"""
    def __init__(self, r: int = 2, embedding_dim: int = 256):
        """
        Initialize the neural model for Tacotron
        Args:
            r: reduction factor
            embedding_dim: dimension of input character embedding
        """
        super(TacotronModel, self).__init__()
        self.r = r
        self.num_mel = AudioProcessParam.n_mels
        self.n_fft_bins = 1 + AudioProcessParam.n_fft // 2  # 1025 by default

        self.encoder = TacotronEncoder(embedding_dim)
        self.decoder = TacotronDecoder(r, self.num_mel)
        self.cbhg = Cbhg(input_emb_dim=self.num_mel,
                         conv_bank_K=8,
                         proj_channel_dims=(256, 80))

        num_cbhg_output_feature = self.cbhg.num_output_features

        # The output of the CBHG must be upsampled to a linear spectrogram using
        # fc layer, which is our final target
        self.fc_lin_spec_target = nn.Linear(
            num_cbhg_output_feature, self.n_fft_bins)

    def get_go_frame(self, batch_size: int) -> torch.Tensor:
        """
        Get <GO> frame, which is a zero vector
        Args:
            batch_size: batch size of input

        Returns:
            <GO> frame vector

        """
        return torch.zeros(batch_size, 1, self.num_mel)

    def forward(self, x: torch.Tensor,
                mel_spec: torch.Tensor) -> TacotronOutput:
        """
        Forward pass for Tacotron at training phase
        Args:
            x: input character embedding. Dim: (batch, seq, feature)
            mel_spec: ground truth mel spectrogram. Dim: (batch, seq, mel)

        Returns:
            predicted linear and mel spectrogram, and attention weight
        """
        encoder_states = self.encoder(x)

        go_frame = self.get_go_frame(x.shape[0]).to(x.device)

        # Input to decoder is <GO> frame and every r-th frame of the ground
        # truth mel spectrogram
        decoder_input = torch.cat(
            [go_frame, mel_spec[:, self.r-1::self.r]], dim=1)
        decoder_output = self.decoder.forward(decoder_input, encoder_states)

        # since prediction from last frame can cause redundant predictions,
        # discard them if needed
        len_residue = self.r - (mel_spec.shape[1] % self.r)
        decoder_output.pred_mel_spec = decoder_output.pred_mel_spec[:, :-len_residue]

        pred_lin_spec = self.fc_lin_spec_target(
            self.cbhg(decoder_output.pred_mel_spec))

        return TacotronOutput(pred_lin_spec=pred_lin_spec,
                              pred_mel_spec=decoder_output.pred_mel_spec,
                              attention_weight=decoder_output.attention_weight)


class TacotronEncoder(nn.Module):
    """
    A Tacotron encoder which incorporates following modules:
        1. a pre-net
        2. a CBHG module
    """

    def __init__(self, embedding_dim: int = 256):
        super(TacotronEncoder, self).__init__()
        self.pre_net = PreNet(num_input_features=embedding_dim,
                              fc_dims=(256, 128))
        self.cbhg = Cbhg(input_emb_dim=128,
                         conv_bank_K=16,
                         proj_channel_dims=(128, 128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tacotron encoder converts a batch of sequences of character into
        a batch of latent vectors of higher level representations.
        The input is assumed to have dimension in form of (batch, seq, feature)
        Args:
            x: embedding representation of input characters

        Returns:
            latent vectors of higher level representations

        """

        return self.cbhg(self.pre_net(x))


class TacotronDecoder(nn.Module):
    """
    A Tacotron decoder which incorporates following modules:
        1. a pre-net
        2. a GRU layer for attention rnn
        3. a residual dual-GRU layer for decoder rnn
        4. a fully connected layer mapping the output of the decoder rnn
            to mel-spectrogram (seq2seq target)
    """

    def __init__(self, r: int = 2, num_mel: int = 80):
        """
        Initiate tacotron decoder
        Args:
            r: reduction factor
            num_mel: number of mel channels for mel spectrogram
        """
        super(TacotronDecoder, self).__init__()
        self.num_mel = num_mel
        self.r = r

        self.attention = ContentBasedAttention(256)

        self.pre_net = PreNet(num_input_features=num_mel,
                              fc_dims=(256, 128))

        self.attention_rnn = nn.GRU(input_size=128,
                                    hidden_size=256,
                                    batch_first=True)

        # 256 from attention RNN and 256 from context vector
        self.decoder_rnn = DecoderRnn(num_input_features=256 + 256,
                                      num_cells=256)

        # The output of the decoder RNN must be transformed into mel-spectrogram
        # which is seq2seq target
        # Two options are available when predicting mel spectrogram
        # 1. Make `r` FC layers mapping 256 -> num_mel
        # 2. Make one FC layer mapping 256 -> r * num_mel and collate data when
        #  making forward pass
        # Use option 2, since it allows information exchange between the r
        # consecutive generated outputs
        self.fc_mel_target = nn.Linear(256, self.r * self.num_mel)

    def forward(self,
                x: torch.Tensor,
                encoder_states: torch.Tensor) -> DecoderOutput:
        """
        Tacotron decoder forward pass
        Args:
            x: mel spectrogram frame(s)
            encoder_states: encoder states from the Tacotron encoder

        Returns:
            dictionary containing predicted mel spectrogram and attention weight

        """
        x = self.attention_rnn(self.pre_net(x))[0]
        attention_output = self.attention.forward(x, encoder_states)

        # Both context vector and network input has dimension of
        # (batch, seq, feature)
        x = self.decoder_rnn(
            torch.cat([x, attention_output.context_vector], dim=-1)
        )

        x = self.fc_mel_target(x)

        # Since each decoder step outputs r frames, we need to collate the
        # output accordingly.
        x_final_shape = (x.shape[0], x.shape[1] * self.r, self.num_mel)
        pred_mel_spec = torch.empty(x_final_shape).to(x.device)
        for r_idx in range(self.r):
            start_idx = r_idx * self.num_mel
            end_idx = start_idx + self.num_mel
            pred_mel_spec[:, r_idx::self.r] = x[:, :, start_idx: end_idx]

        return DecoderOutput(pred_mel_spec, attention_output.attention_weight)
