from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from typing import Union, List, Tuple

import librosa
import numpy as np
import scipy.signal


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
    min_level_db: int = -100
    ref_level_db: int = 20


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
            y = cls.pre_emphasize(y)
        return y

    @classmethod
    def pre_emphasize(cls, x: np.ndarray, emphasis_factor: float = 0.97):
        return scipy.signal.lfilter([1, -emphasis_factor], [1], x)

    @classmethod
    def de_emphasize(cls, x: np.ndarray, emphasis_factor: float = 0.97):
        return scipy.signal.lfilter([1], [1, -emphasis_factor], x)

    @classmethod
    def audio2spec(cls, y: np.ndarray) -> np.ndarray:
        """Convert a PCM data into magnitude spectrogram"""
        lin_stft = librosa.stft(y, n_fft=cls.param.n_fft,
                                hop_length=cls.param.hop_length,
                                window=cls.param.window_type)
        return np.abs(lin_stft)

    @classmethod
    def lin2db(cls, spec: np.ndarray) -> np.ndarray:
        """Converts linear spectrogram into log spectrogram. Clip the
        spectrogram value less than 1e-7 to avoid taking log of zero"""
        return 20 * np.log10(np.clip(spec, a_min=1e-5, a_max=None))

    @classmethod
    def spec2mel_spec(cls, spec: np.ndarray) -> np.ndarray:
        """Converts magnitude spectrogram into mel spectrogram"""
        # power of 2, as the mel filter maps into power spectrum
        return np.dot(cls.mel_filter, spec ** 2)

    @classmethod
    def db2lin(cls, log_spec: np.ndarray) -> np.ndarray:
        """Converts log spectrogram into spectrogram"""
        return np.power(10.0, log_spec / 20)

    @classmethod
    def spec2audio(cls, spec: np.ndarray, n_iter: int = 60,
                   enhancement_factor: float = 1.5) -> np.ndarray:
        """Converts magnitude spectrogram into a waveform using Griffin-Lim
        algorithm and istft"""
        spec = cls.denormalize(spec)
        spec = cls.db2lin(spec + AudioProcessParam.ref_level_db)
        spec = np.power(spec, enhancement_factor)  # enhance
        waveform = librosa.griffinlim(spec, n_iter=n_iter,
                                      hop_length=cls.param.hop_length)
        # TODO: de-emphasize the waveform?
        return cls.de_emphasize(waveform)

    @classmethod
    def normalize(cls, spec: np.ndarray):
        spec = spec - AudioProcessParam.min_level_db
        spec = spec / - AudioProcessParam.min_level_db
        return np.clip(spec, 0, 1)

    @classmethod
    def denormalize(cls, spec: np.ndarray):
        spec = np.clip(spec, 0, 1) * -AudioProcessParam.min_level_db
        return spec + AudioProcessParam.min_level_db

    @classmethod
    def audio_file_to_specs(cls, audio_file: str,
                            pre_emphasize: bool = True) -> SpecData:
        """Create mel and log spectrogram from an audio file"""
        id = os.path.splitext(os.path.basename(audio_file))[0]

        y = cls.load_audio(audio_file, pre_emphasize)
        spec = cls.audio2spec(y)

        log_lin_spec = cls.lin2db(spec) - AudioProcessParam.ref_level_db
        log_lin_spec = cls.normalize(log_lin_spec)

        mel_spec = cls.spec2mel_spec(spec)
        log_mel_spec = cls.lin2db(mel_spec) - AudioProcessParam.ref_level_db
        log_mel_spec = cls.normalize(log_mel_spec)

        return SpecData(id, log_lin_spec, log_mel_spec)
