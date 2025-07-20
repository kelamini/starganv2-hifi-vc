#coding: utf-8

import os.path as osp
import random
import numpy as np
import random
import soundfile as sf
from scipy.io.wavfile import read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from g2p_en import G2p

from configs import config

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from text_utils import TextCleaner

np.random.seed(1)
random.seed(1)

DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'word_index_dict.txt')
SEGMENT_SIZE = config['preprocess_params']['segment_size']
SPECT_PARAMS = config['preprocess_params']['spect_params']
MEL_PARAMS = config['preprocess_params']['mel_params']
MAX_WAV_VALUE = 32768.0


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
                 dict_path=DEFAULT_DICT_PATH,
                 sr=24000
                ):
        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.dict_path = dict_path
        self.sr = sr
        self._g2p = None
        self._text_cleaner = None

    @property
    def g2p(self):
        if self._g2p is None:
            from g2p_en import G2p
            self._g2p = G2p()
        return self._g2p

    @property
    def text_cleaner(self):
        if self._text_cleaner is None:
            from text_utils import TextCleaner
            self._text_cleaner = TextCleaner(self.dict_path)
        return self._text_cleaner

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        wave, text_tensor, speaker_id = self._load_tensor(data)
        wave_tensor = torch.from_numpy(wave).float()
        wave_tensor_mel = self._load_tensor_mel(data)
        mel_tensor = mel_spectrogram(wave_tensor_mel)
        length_feature = mel_tensor.size(1)

        if (text_tensor.size(0)+1) >= (mel_tensor.size(1) // 3):
            mel_tensor = F.interpolate(
                mel_tensor.unsqueeze(0), size=(text_tensor.size(0)+1)*3, align_corners=False,
                mode='linear').squeeze(0)

        acoustic_feature = mel_tensor[:, :(length_feature - length_feature % 2)]

        return wave_tensor, acoustic_feature, text_tensor, data[0]

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(wave_path)

        # phonemize the text
        ps = self.g2p(text.replace('-', ' '))
        if "'" in ps:
            ps.remove("'")
        text = self.text_cleaner(ps)
        blank_index = self.text_cleaner.word_index_dictionary[" "]
        text.insert(0, blank_index) # add a blank at the beginning (silence)
        text.append(blank_index) # add a blank at the end (silence)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_tensor_mel(self, data):
        wave_path, text, speaker_id = data
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
      return_wave (bool): if true, will return the wave data along with spectrogram. 
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        paths = ['' for _ in range(batch_size)]
        for bid, (_, mel, text, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            assert(text_size < (mel_size//2))

        if self.return_wave:
            waves = [b[0] for b in batch]
            return texts, input_lengths, mels, output_lengths, paths, waves

        return texts, input_lengths, mels, output_lengths


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
