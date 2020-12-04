from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from typing import Union, List, Tuple

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioProcessParam:
    """Parameter required for audio processing including loading speech audio,
    pre-emphasis, and constructing linear and mel spectrogram"""
    sr: int = 22050
    pre_emph_coef: float = 0.97
    n_fft: int = 2048
    hop_length: int = 256
    window_type: int = 'hann'
    n_mels: int = 80


@dataclass
class SpecData:
    """Input required for training Tacotron which are log spectrogram and
    mel spectrogram"""
    id: str
    log_lin_spec: np.ndarray
    log_mel_spec: np.ndarray


class AudioProcessingHelper:
    param = AudioProcessParam()
    mel_filter = librosa.filters.mel(sr=param.sr,
                                     n_fft=param.n_fft,
                                     n_mels=param.n_mels)

    @classmethod
    def load_audio(cls, audio_file: Union[Path, str],
                   pre_emphasize: bool = True) -> np.ndarray:
        """Load audio from an audio file. Pre-emphasize"""
        y, sr = librosa.load(audio_file, sr=cls.param.sr, mono=True)
        if pre_emphasize:
            y = librosa.effects.preemphasis(y, coef=cls.param.pre_emph_coef)
        return y

    @classmethod
    def audio2spec(cls, y: np.ndarray) -> np.ndarray:
        """Convert a PCM data into spectrogram"""
        lin_stft = librosa.stft(y, n_fft=cls.param.n_fft,
                                hop_length=cls.param.hop_length,
                                window=cls.param.window_type)
        return np.abs(lin_stft) ** 2

    @classmethod
    def lin2log(cls, spec: np.ndarray) -> np.ndarray:
        """Converts linear spectrogram into log spectrogram. Clip the
        spectrogram value less than 1e-7 to avoid taking log of zero"""
        return np.log10(np.clip(spec, a_min=1e-7, a_max=None))

    @classmethod
    def spec2mel_spec(cls, spec: np.ndarray) -> np.ndarray:
        """Converts spectrogram into mel spectrogram"""
        return np.dot(cls.mel_filter, spec)

    @classmethod
    def log2lin(cls, log_spec: np.ndarray) -> np.ndarray:
        """Converts log spectrogram into spectrogram"""
        return np.power(10, log_spec)

    @classmethod
    def spec2audio(cls, spec: np.ndarray, n_iter: int = 60,
                   enhancement_factor: float = 1.5,
                   normalize: bool = True) -> np.ndarray:
        """Converts magnitude spectrogram into a waveform using Griffin-Lim
        algorithm and istft"""
        stft = np.power(spec, 0.5)  # Power to magnitude
        stft = np.power(stft, enhancement_factor)  # enhance
        waveform = librosa.griffinlim(stft, n_iter=n_iter,
                                      hop_length=cls.param.hop_length)
        if normalize:
            waveform = waveform / max(0.01, np.max(np.abs(waveform)))

        return waveform

    @classmethod
    def audio_file_to_specs(cls, audio_file: str,
                            pre_emphasize: bool = True) -> SpecData:
        """Create mel and log spectrogram from an audio file"""
        id = os.path.splitext(os.path.basename(audio_file))[0]

        y = cls.load_audio(audio_file, pre_emphasize)
        spec = cls.audio2spec(y)

        log_lin_spec = cls.lin2log(spec)
        mel_spec = cls.spec2mel_spec(spec)
        log_mel_spec = cls.lin2log(mel_spec)

        return SpecData(id, log_lin_spec, log_mel_spec)
