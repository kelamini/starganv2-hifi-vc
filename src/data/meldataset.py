#coding: utf-8
import random

import numpy as np
import soundfile as sf
from scipy.io.wavfile import read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import torch
import torchaudio
from torch.utils.data import DataLoader

from src.configs import config

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
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)

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
                 validation=False,
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(d[0], int(d[-1])) for d in _data_list]
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target] \
            for target in list(set([label for _, label in self.data_list]))}

        self.sr = sr
        self.validation = validation
        self.max_mel_length = SEGMENT_SIZE // MEL_PARAMS['hop_length']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, label = self._load_data(data)
        ref_data = random.choice(self.data_list)
        ref_mel_tensor, ref_label = self._load_data(ref_data)
        ref2_data = random.choice(self.data_list_per_class[ref_label])
        ref2_mel_tensor, _ = self._load_data(ref2_data)
        return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label
    
    def _load_data(self, path):
        wave_tensor, label = self._load_tensor(path)
        
        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor
        
        mel_tensor = mel_spectrogram(wave_tensor)

        return mel_tensor, label

    def _preprocess(self, wave_tensor, ):
        mel_tensor = mel_spectrogram(wave_tensor)
        return mel_tensor

    def _load_tensor(self, data):
        wave_path, label = data
        label = int(label)
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

        return audio, label


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = SEGMENT_SIZE // MEL_PARAMS['hop_length']
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()

        for bid, (mel, label, ref_mel, ref2_mel, ref_label) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel
            
            labels[bid] = label
            ref_labels[bid] = ref_label

        z_trg = torch.randn(batch_size, self.latent_dim)
        z_trg2 = torch.randn(batch_size, self.latent_dim)
        
        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, validation=validation)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=True,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
