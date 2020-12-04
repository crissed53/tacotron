import multiprocessing as mp
import os
from typing import List

import numpy as np
import tqdm

from data import LIN_SPEC_SAVE_DIR, MEL_SPEC_SAVE_DIR
from data.audio import AudioProcessingHelper


def get_audio_list(audio_root: str) -> List[str]:
    """Get list of audio files in audio_root"""
    return [filename for filename in os.listdir(audio_root)
            if os.path.splitext(filename)[1] == '.wav']


def generate_from_audiofile(audio_file: str):
    """Generate a pair of mel and log spectrogram from a single audio file"""
    input_spec = AudioProcessingHelper.audio_file_to_specs(audio_file)

    # save log spectrogram
    np.save(os.path.join(LIN_SPEC_SAVE_DIR, input_spec.id),
            input_spec.log_lin_spec)

    # save mel spectrogram
    np.save(os.path.join(MEL_SPEC_SAVE_DIR, input_spec.id),
            input_spec.log_mel_spec)


def generate_lj_speech_specs(audio_root: str,
                             num_workers: int = 1) -> None:
    """
    Generate spectrograms required for LJ Speech dataset. The directories
    in which the spectrograms are saved can be described as follows:

    dataset
        |- mel_spec
            |- LJ001-0001.npy
            |- LJ001-0002.npy
            |- LJ001-0003.npy
            ...

        |- lin_spec
            |- LJ001-0001.npy
            |- LJ001-0002.npy
            |- LJ001-0003.npy
            ...

    Args:
        audio_root: root directory in which audio files for the LJ Speech
            dataset are saved
        num_workers: number of processes to use when generating the spectrograms
    """

    if not os.path.exists(LIN_SPEC_SAVE_DIR):
        os.makedirs(LIN_SPEC_SAVE_DIR)

    if not os.path.exists(MEL_SPEC_SAVE_DIR):
        os.makedirs(MEL_SPEC_SAVE_DIR)

    audio_list = [os.path.join(audio_root, filename)
                  for filename in get_audio_list(audio_root)]

    with mp.Pool(num_workers) as pool:
        for _ in tqdm.tqdm(
                pool.imap_unordered(generate_from_audiofile, audio_list),
                total=len(audio_list),
                desc='Generating Spectrogram for LJSpeech'):
            pass


if __name__ == '__main__':
    from argparse import ArgumentParser

    p = ArgumentParser()
    p.add_argument('--audio_root', type=str,
                   default=os.path.join('dataset', 'LJSpeech-1.1', 'wavs'))
    p.add_argument('--num_workers', type=int, default=1)
    args = p.parse_args()

    generate_lj_speech_specs(audio_root=args.audio_root,
                             num_workers=args.num_workers)
