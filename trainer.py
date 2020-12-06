from dataclasses import dataclass
import io
import os
from typing import Dict, List

from data import METADATA_FILE
from data.audio import AudioProcessingHelper, AudioProcessParam
from data.text import tokenize_transcription
from data.torch import TorchLJSpeechDataset, TorchLJSpeechBatch, TorchLJSpeechData
from model.tacotron import Tacotron
import PIL.Image

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, LambdaLR, MultiStepLR
from torch.utils.data import DataLoader, random_split, Subset
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

plt.ioff()  # turn off interactive plotting


class TacotronLoss(nn.Module):
    def __init__(self, mel_recon_coeff: float = 0.5, lin_recon_coeff: float = 0.5):
        super(TacotronLoss, self).__init__()
        self.mel_recon_coeff = mel_recon_coeff
        self.lin_recon_coeff = lin_recon_coeff
        self.l1_loss = nn.L1Loss()

    def forward(self,
                pred_mel_spec: torch.Tensor,
                truth_mel_spec: torch.Tensor,
                pred_lin_spec: torch.Tensor,
                truth_lin_spec: torch.Tensor):
        """
        Forward pass for calculating loss for Tacotron
        Args:
            pred_mel_spec: predicted mel spectrogram
            truth_mel_spec: ground truth mel spectrogram
            pred_lin_spec: predicted log spectrogram
            truth_lin_spec: ground truth log spectrogram

        Returns:
            loss function

        """
        return (self.mel_recon_coeff * self.l1_loss(pred_mel_spec, truth_mel_spec)
                + self.lin_recon_coeff * self.l1_loss(pred_lin_spec, truth_lin_spec))


@dataclass
class SampleResult:
    uid: str
    transcription: str
    truth_lin_spec: np.ndarray
    pred_lin_spec: np.ndarray
    truth_mel_spec: np.ndarray
    pred_mel_spec: np.ndarray
    attention_weight: np.ndarray
    truth_audio: np.ndarray
    pred_audio: np.ndarray


class TacotronTrainer:
    TRAIN_STAGE = 'train'
    VAL_STAGE = 'val'
    VERSION_FORMAT = 'VERSION_{}'
    MODEL_SAVE_FORMAT = 'version_{version:03}_model_{epoch:04}.pth'

    def __init__(self,
                 batch_size: int = 32,
                 num_epoch: int = 100,
                 train_split: float = 0.9,
                 log_interval: int = 1000,
                 log_audio_factor: int = 5,
                 num_data: int = None,
                 log_root: str = './tb_logs',
                 save_root: str = './checkpoints',
                 num_workers: int = 4,
                 version: int = None,
                 num_test_samples: int = 5):
        """
        Initialize tacotron trainer
        Args:
            batch_size: batch size
            num_epoch: total number of epochs to train
            train_split: train ratio of train-val split
            log_interval: interval for test sample logging to tensorboard in
                epoch unit
            log_audio_factor: number of log_interval for logging audio
                which requires quite a lot of overhead
            num_data: number of datapoints to load in the dataset
            log_root: root directory for the tensorboard logging
            save_root: root directory for saving model
            num_workers: number of workers for dataloader
            version: version of training
            num_test_samples: number of test samples to generate
                for each logging
        """
        if not os.path.exists(log_root):
            os.makedirs(log_root)

        if not os.path.exists(save_root):
            os.makedirs(save_root)

        is_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if is_cuda else 'cpu')

        self.train_split = train_split

        self.epoch_num = num_epoch

        self.splitted_dataset = self.__split_dataset(
            TorchLJSpeechDataset(num_data=num_data))

        self.dataloaders = self.__get_dataloaders(
            batch_size, num_workers=num_workers)

        self.tacotron = Tacotron()
        self.tacotron.to(self.device)

        self.loss = TacotronLoss()
        self.optimizer = Adam(self.tacotron.nn.parameters())
        self.lr_scheduler = StepLR(
            optimizer=self.optimizer,
            step_size=10000,
            gamma=0.9)

        if version is None:
            versions = os.listdir(log_root)
            if not versions:
                self.version = 0
            else:
                self.version = max([int(ver[-1]) for ver in versions]) + 1

        log_dir = os.path.join(
            log_root, self.VERSION_FORMAT.format(self.version))
        if os.path.exists(log_dir):
            os.remove(log_dir)

        self.logger = SummaryWriter(log_dir)

        self.save_root = save_root

        self.log_interval = log_interval
        self.log_audio_factor = log_audio_factor

        self.global_step = 0
        self.running_count = {self.TRAIN_STAGE: 0,
                              self.VAL_STAGE: 0}
        self.running_loss = {self.TRAIN_STAGE: 0,
                             self.VAL_STAGE: 0}

        self.sample_indices = list(range(num_test_samples))

    def fit(self):
        for epoch in tqdm.tqdm(range(self.epoch_num),
                               total=self.epoch_num,
                               desc='Epoch'):
            self.__run_epoch(epoch)

    def __run_epoch(self, epoch: int):
        # reset running loss and count after each epoch
        self.__reset_loss()
        self.__reset_count()

        for stage, dataloader in self.dataloaders.items():
            prog_bar = tqdm.tqdm(dataloader,
                                 desc=f'{stage.capitalize()} in progress',
                                 total=len(dataloader))
            for batch in dataloader:
                self.__run_step(batch, stage, prog_bar)

        # epoch vs global step
        self.logger.add_scalar('epoch', epoch, global_step=self.global_step)

        # add loss to logger
        loss_dict = {stage: self.__calculate_mean_loss(stage)
                     for stage in self.running_loss}
        self.logger.add_scalars('loss', loss_dict, global_step=epoch)

        # save model for each
        save_file = os.path.join(
            self.save_root,
            self.MODEL_SAVE_FORMAT.format(version=self.version, epoch=epoch)
        )
        self.tacotron.save(save_file, self.device)

    def __run_step(self, batch: TorchLJSpeechBatch, stage: str,
                   prog_bar: tqdm.tqdm):
        if stage == self.TRAIN_STAGE:
            self.tacotron.nn.train()
            self.optimizer.zero_grad()
        else:
            self.tacotron.nn.eval()

        batch = batch.to(self.device)

        output = self.tacotron.forward_train(batch)
        loss_val = self.loss(batch.mel_spec, output.pred_mel_spec,
                             batch.lin_spec, output.pred_lin_spec)

        self.running_loss[stage] += loss_val.item() * batch.mel_spec.size(0)
        self.running_count[stage] += batch.mel_spec.size(0)

        if stage == self.TRAIN_STAGE:
            loss_val.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                self.logger.add_scalar('training_loss',
                                       self.__calculate_mean_loss(stage),
                                       global_step=self.global_step)
                log_audio = False
                if self.global_step % (self.log_interval * self.log_audio_factor) == 0:
                    log_audio = True
                sample_results = self.__get_sample_results()
                for sample_result in sample_results:
                    self.__log_sample_results(
                        self.global_step, sample_result, log_audio=log_audio)

        prog_bar.update()
        prog_bar.set_postfix(
            {'Running Loss': f'{self.__calculate_mean_loss(stage):.3f}'})

    def __log_sample_results(self, steps: int,
                             sample_result: SampleResult,
                             log_mel: bool = True,
                             log_spec: bool = True,
                             log_attention: bool = True,
                             log_audio: bool = True) -> None:
        """
        Log the sample results into tensorboard
        Args:
            steps: current step
            sample_result: sample result to log
            log_mel: if True, log mel spectrogram
            log_spec: if True, log spectrogram
            log_attention: if True, log attention
            log_audio: if True, log audio

        """
        if log_mel:
            title = f'Log Mel Spectrogram, Step:{steps}, ' \
                    f'Uid: {sample_result.uid}'

            fig = self.__get_spec_plot(
                pred_spec=sample_result.pred_mel_spec,
                truth_spec=sample_result.truth_mel_spec,
                suptitle=title,
                ylabel='Mel')
            img_tensor = self.__get_plot_tensor(fig)
            tag = f'mel_spec/{sample_result.uid}'
            self.logger.add_image(tag, img_tensor, global_step=steps)

        if log_spec:
            title = f'Log Spectrogram, Step:{steps}, ' \
                    f'Uid: {sample_result.uid}'
            fig = self.__get_spec_plot(
                pred_spec=sample_result.pred_lin_spec,
                truth_spec=sample_result.truth_lin_spec,
                suptitle=title,
                ylabel='DFT bins')
            img_tensor = self.__get_plot_tensor(fig)
            tag = f'lin_spec/{sample_result.uid}'
            self.logger.add_image(tag, img_tensor, global_step=steps)

        if log_attention:
            title = f'Attention Weight, Epoch :{steps}, ' \
                    f'Uid: {sample_result.uid}'
            fig = self.__get_attention_plot(
                title=title,
                attention_weight=sample_result.attention_weight)
            img_tensor = self.__get_plot_tensor(fig)
            tag = f'attention/{sample_result.uid}'
            self.logger.add_image(tag, img_tensor, global_step=steps)

        if log_audio:
            pred_tag = f'audio/{sample_result.uid}_predicted'
            truth_tag = f'audio/{sample_result.uid}_truth'

            self.logger.add_audio(
                tag=pred_tag,
                snd_tensor=torch.from_numpy(
                    sample_result.pred_audio).unsqueeze(1),  # add channel dim
                global_step=steps,
                sample_rate=AudioProcessParam.sr
            )

            self.logger.add_audio(
                tag=truth_tag,
                snd_tensor=torch.from_numpy(
                    sample_result.truth_audio).unsqueeze(1),  # add channel dim
                global_step=steps,
                sample_rate=AudioProcessParam.sr
            )

    def __get_sample_results(self) -> List[SampleResult]:
        """
        Get sample results to show in tensorboard, including
            1. Predicted and ground truth spectrogram pairs
            2. Predicted and ground truth mel spectrogram pairs
            3. Predicted and ground truth audio pairs
            4. Attention weight
        Returns:
            list of sample results

        """
        val_dataset = self.splitted_dataset[self.VAL_STAGE]
        self.tacotron.nn.eval()

        test_insts = []
        with torch.no_grad():
            for subset_i in self.sample_indices:
                datapoint: TorchLJSpeechData = val_dataset[subset_i]
                datapoint: TorchLJSpeechBatch = datapoint.add_batch_dim()
                datapoint = datapoint.to(self.device)

                ds_idx = val_dataset.indices[subset_i]
                uid = val_dataset.dataset.uids[ds_idx]

                # Transcription
                transcription = val_dataset.dataset.uid_to_transcription[uid]

                wav_filepath = os.path.join(
                    val_dataset.dataset.wav_save_dir, f'{uid}.wav')
                truth_audio = AudioProcessingHelper.load_audio(wav_filepath)

                taco_output = self.tacotron.forward_train(datapoint)

                spec = taco_output.pred_lin_spec.squeeze(0).cpu().numpy().T
                pred_audio = AudioProcessingHelper.spec2audio(spec)

                test_insts.append(
                    SampleResult(
                        uid=uid,
                        transcription=transcription,
                        truth_lin_spec=datapoint.lin_spec.squeeze(0).cpu().numpy().T,
                        pred_lin_spec=taco_output.pred_lin_spec.squeeze(0).cpu().numpy().T,
                        truth_mel_spec=datapoint.mel_spec.squeeze(0).cpu().numpy().T,
                        pred_mel_spec=taco_output.pred_mel_spec.squeeze(0).cpu().numpy().T,
                        attention_weight=taco_output.attention_weight.squeeze(0).cpu().numpy(),
                        truth_audio=truth_audio,
                        pred_audio=pred_audio
                    )
                )

        return test_insts

    @staticmethod
    def __get_attention_plot(
            title: str, attention_weight: np.ndarray) -> plt.Figure:
        """
        Get figure handle for attention plot

        Args:
            title: title of the plot
            attention_weight: attention weight to plot

        Returns:
            figure object

        """
        fig = plt.figure(figsize=(6, 5), dpi=80)
        plt.title(title)
        plt.imshow(attention_weight, aspect='auto')
        plt.colorbar()
        plt.xlabel('Encoder seq')
        plt.ylabel('Decoder seq')
        plt.gca().invert_yaxis()  # Let the x, y axis start from the left-bottom corner
        plt.close(fig)
        return fig

    @staticmethod
    def __get_spec_plot(pred_spec: np.ndarray, truth_spec: np.ndarray,
                        suptitle: str, ylabel: str) -> plt.Figure:
        """
        Get a juxtaposition two spectrograms with appropriate title
        Args:
            pred_spec: predicted spectrogram
            truth_spec: ground truth spectrogram
            suptitle: title of the plot
            ylabel: unit of frequency axis of the spectrograms

        Returns:
            figure object

        """
        vmin = min(np.min(truth_spec), np.min(pred_spec))
        vmax = max(np.max(truth_spec), np.max(pred_spec))

        fig = plt.figure(figsize=(11, 5), dpi=80)
        plt.suptitle(suptitle)

        ax1 = plt.subplot(121)
        plt.title('Ground Truth')
        plt.xlabel('Frame')
        plt.ylabel(ylabel)
        plt.imshow(truth_spec, vmin=vmin, vmax=vmax, aspect='auto')
        plt.gca().invert_yaxis()  # let the x, y axis start from the left-bottom corner

        ax2 = plt.subplot(122)
        plt.title('Predicted')
        plt.xlabel('Frame')
        im = plt.imshow(pred_spec, vmin=vmin, vmax=vmax, aspect='auto')
        plt.gca().invert_yaxis()  # let the x, y axis start from the left-bottom corner

        fig.tight_layout()
        fig.colorbar(im, ax=[ax1, ax2])
        plt.close(fig)

        return fig

    @staticmethod
    def __get_plot_tensor(fig) -> torch.Tensor:
        """
        Get tensor for the given figure object
        Args:
            fig: the figure object to convert into tensor

        Returns:
            tensor of the figure

        """
        buf = io.BytesIO()
        fig.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        return image

    def __calculate_mean_loss(self, stage: str) -> float:
        """
        Calculate mean loss for given stage (train/val)
        Args:
            stage: train/val

        Returns:
            mean loss

        """
        return self.running_loss[stage] / self.running_count[stage]

    def __reset_loss(self) -> None:
        self.running_loss = {self.TRAIN_STAGE: 0,
                             self.VAL_STAGE: 0}

    def __reset_count(self) -> None:
        self.running_count = {self.TRAIN_STAGE: 0,
                              self.VAL_STAGE: 0}

    def __split_dataset(self, dataset: TorchLJSpeechDataset) -> Dict[str, Subset]:
        """
        Split the dataset into train/validation set
        Args:
            dataset: dataset to split

        Returns:
            splitted dataset

        """
        num_train_data = int(len(dataset) * self.train_split)
        num_val_data = len(dataset) - num_train_data
        train_dataset, val_dataset = random_split(
            dataset, [num_train_data, num_val_data])

        return {self.TRAIN_STAGE: train_dataset,
                self.VAL_STAGE: val_dataset}

    def __get_dataloaders(
            self, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
        return {stage: DataLoader(
            dataset, shuffle=(stage == self.TRAIN_STAGE),
            collate_fn=TorchLJSpeechDataset.batch_tacotron_input,
            pin_memory=True, batch_size=batch_size,
            num_workers=num_workers)
            for stage, dataset in self.splitted_dataset.items()
        }


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--log_interval', type=int, default=1000)
    p.add_argument('--num_epoch', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=32)
    args = p.parse_args()

    trainer = TacotronTrainer(num_workers=args.num_workers,
                              log_interval=args.log_interval,
                              num_epoch=args.num_epoch,
                              batch_size=args.batch_size)
    # trainer = TacotronTrainer(num_data=32*10, log_interval=10, log_audio_factor=1, num_workers=args.num_workers)
    trainer.fit()
