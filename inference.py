import os
from pathlib import Path
import argparse
import json

import yaml
from munch import Munch

import torch
import librosa
from librosa.util import normalize
from scipy.io.wavfile import write, read

from src.models.jdc.model import JDCNet
from src.models.vc.models import Generator, MappingNetwork, StyleEncoder
from src.data.meldataset import mel_spectrogram

from third_party.hifigan.env import AttrDict
from third_party.hifigan.meldataset import MAX_WAV_VALUE
from third_party.hifigan.models import Generator


torch.manual_seed(42)
global device
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def _load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def _load_hifigan_configs(config_path):
    with open(config_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    return AttrDict(data)


def load_hifigan_model(model_path):
    configs = _load_hifigan_configs(os.path.join(os.path.dirname(model_path), "config.json"))
    generator = Generator(configs).to(device)

    state_dict_g = _load_checkpoint(model_path, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.eval()
    generator.remove_weight_norm()

    return generator


def hifigan_inference(generator, mel):
    with torch.no_grad():
        y_g_hat = generator(mel)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
    
    return audio


def _build_starganv2_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema


def load_starganv2_model(model_path):
    config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        starganv2_config = yaml.safe_load(f)
    starganv2 = _build_starganv2_model(model_params=starganv2_config["model_params"])
    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']
    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    starganv2.mapping_network = starganv2.mapping_network.to('cuda')
    starganv2.generator = starganv2.generator.to('cuda')

    return starganv2


def load_f0_model(model_path):
    f0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(model_path)['net']
    f0_model.load_state_dict(params)
    _ = f0_model.eval()
    f0_model = f0_model.to('cuda')
    
    return f0_model


def compute_style(starganv2, path, speaker: int):
    if path == "":
        label = torch.LongTensor([speaker]).to('cuda')
        latent_dim = starganv2.mapping_network.shared[0].in_features
        ref = starganv2.mapping_network(torch.randn(1, latent_dim).to('cuda'), label)
    else:
        wave, sr = load_wav(path)
        if sr != 44100:
            wave = librosa.resample(wave, sr, 44100)
        mel_tensor = wave2mel(wave).to('cuda')

        with torch.no_grad():
            label = torch.LongTensor([speaker])
            ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
    
    return ref


def wave2mel(audio):
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    mel_tensor = mel_spectrogram(audio)

    return mel_tensor


def inference(args):
    f0_model = load_f0_model(args.f0_model_path)
    starganv2 = load_starganv2_model(args.starganv2_vc_model_path)
    hifigan = load_hifigan_model(args.hifigan_model_path)

    audio, sr = load_wav(args.source_path)
    source_mel = wave2mel(audio).to(device)

    with torch.no_grad():
        ref = compute_style(starganv2, args.ref_path, args.speaker)
        f0_feat = f0_model.get_feature_GAN(source_mel.unsqueeze(1))
        gen_mel = starganv2.generator(source_mel.unsqueeze(1), ref, F0=f0_feat)

        gen_mel = gen_mel.transpose(-1, -2).squeeze().to(device)
        audio = hifigan_inference(hifigan, gen_mel)
    
    output_path = os.path.join(args.output_dir, Path(args.source_path).stem+"_vc_generated.wav")
    write(output_path, sr, audio)
    print(f"保存到路径：{args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--f0_model_path', type=str, default='ckpts/jdc/bst.t7', help='')
    parser.add_argument('--starganv2_vc_model_path', type=str, default='', help='')
    parser.add_argument('--hifigan_model_path', type=str, default='ckpts/hifigan/g_00082000', help='')
    parser.add_argument('--source_path', type=str, default='', help='')
    parser.add_argument('--ref_path', type=str, default='', help='')
    parser.add_argument('--speaker', type=int, default=0, help='')
    parser.add_argument('--output_dir', type=str, default='outputs/inference/vc', help='')

    args = parser.parse_args()

    inference(args)
