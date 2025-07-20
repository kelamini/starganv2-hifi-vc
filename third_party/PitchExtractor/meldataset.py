#coding: utf-8
"""
TODO:
- make TestDataset
- separate transforms
"""

import os
import random

import numpy as np
import soundfile as sf
from scipy.io.wavfile import read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import torch
import torchaudio
from torch.utils.data import DataLoader

import pyworld as pw

from configs import config

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

MAX_WAV_VALUE = 32768.0
SEGMENT_SIZE = config['preprocess_params']['segment_size']
SPECT_PARAMS = config['preprocess_params']['spect_params']
MEL_PARAMS = config['preprocess_params']['mel_params']


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram(
        y, 
        n_fft=MEL_PARAMS['n_fft'], 
        num_mels=MEL_PARAMS['n_mels'], 
        sampling_rate=config['preprocess_params']['sr'], 
        hop_size=MEL_PARAMS['hop_length'], 
        win_size=MEL_PARAMS['win_length'], 
        fmin=MEL_PARAMS['f_min'], 
        fmax=MEL_PARAMS['f_max'], 
        center=False,
    ):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=torch.hann_window(win_size).to(y.device),
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(torch.from_numpy(mel).float().to(y.device), spec)
    spec = spectral_normalize_torch(spec)

    return spec.squeeze(0)


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 verbose=True
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [d[0] for d in _data_list]

        self.sr = sr

        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = SEGMENT_SIZE // MEL_PARAMS['hop_length']
        
        self.verbose = verbose
        
        # for silence detection
        self.zero_value = -10 # what the zero value is
        self.bad_F0 = 5 # if less than 5 frames are non-zero, it's a bad F0, try another algorithm

    def __len__(self):
        return len(self.data_list)

    def path_to_mel_and_label(self, path):
        wave_tensor_f0 = self._load_tensor_f0(path)
        wave_tensor_mel = self._load_tensor_mel(path)
        
        # use pyworld to get F0
        output_file = path + "_f0.npy"
        # check if the file exists
        if os.path.isfile(output_file): # if exists, load it directly
            f0 = np.load(output_file)
        else: # if not exist, create F0 file
            if self.verbose:
                print('Computing F0 for ' + path + '...')
            x = wave_tensor_f0.numpy().astype("double")
            frame_period = MEL_PARAMS['hop_length'] * 1000 / self.sr
            _f0, t = pw.harvest(x, self.sr, frame_period=frame_period)
            if sum(_f0 != 0) < self.bad_F0: # this happens when the algorithm fails
                _f0, t = pw.dio(x, self.sr, frame_period=frame_period) # if harvest fails, try dio
            f0 = pw.stonemask(x, _f0, t, self.sr)
            # save the f0 info for later use
            np.save(output_file, f0)
        
        f0 = torch.from_numpy(f0).float()
        f0_length = f0.size(0)
        
        if self.data_augmentation:
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor_mel = random_scale * wave_tensor_mel

        mel_tensor = mel_spectrogram(wave_tensor_mel)

        f0_zero = (f0 == 0)
        
        #######################################
        # You may want your own silence labels here
        # The more accurate the label, the better the resultss
        is_silence = torch.zeros(f0.shape)
        is_silence[f0_zero] = 1
        #######################################
        
        if f0_length > self.max_mel_length:
            random_start = np.random.randint(0, f0_length - self.max_mel_length)
            f0 = f0[random_start:random_start + self.max_mel_length]
            is_silence = is_silence[random_start:random_start + self.max_mel_length]
        
        if torch.any(torch.isnan(f0)): # failed
            f0[torch.isnan(f0)] = self.zero_value # replace nan value with 0
        
        return mel_tensor, f0, is_silence


    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, f0, is_silence = self.path_to_mel_and_label(data)
        return mel_tensor, f0, is_silence

    def _load_tensor_f0(self, data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

    def _load_tensor_mel(self, data):
        wave_path = data
        audio, sampling_rate = load_wav(wave_path)

        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if audio.size(1) >= SEGMENT_SIZE:
            max_audio_start = audio.size(1) - SEGMENT_SIZE
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+SEGMENT_SIZE]
        else:
            audio = torch.nn.functional.pad(audio, (0, SEGMENT_SIZE - audio.size(1)), 'constant')

        return audio


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.min_mel_length = SEGMENT_SIZE // MEL_PARAMS['hop_length']
        self.max_mel_length = SEGMENT_SIZE // MEL_PARAMS['hop_length']
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        f0s = torch.zeros((batch_size, self.max_mel_length)).float()
        is_silences = torch.zeros((batch_size, self.max_mel_length)).float()

        for bid, (mel, f0, is_silence) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            f0s[bid, :mel_size] = f0
            is_silences[bid, :mel_size] = is_silence

        if self.max_mel_length > self.min_mel_length:
            random_slice = np.random.randint(
                self.min_mel_length//self.mel_length_step,
                1+self.max_mel_length//self.mel_length_step) * self.mel_length_step + self.min_mel_length
            mels = mels[:, :, :random_slice]
            f0 = f0[:, :random_slice]

        mels = mels.unsqueeze(1)
        return mels, f0s, is_silences


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):
    
    dataset = MelDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
