# StarGANv2-HiFi-VC

## 该项目继承自 [StarGANv2-VC](https://github.com/yl4579/StarGANv2-VC.git)


[说明：项目还在构建，施工中...]


## 环境准备
1. Python >= 3.10
2. 克隆仓库:
```bash
git clone https://github.com/kelamini/starganv2-hifi-vc.git
cd starganv2-hifi-vc
```
3. 创建环境: 
```bash
conda create -n starganv2-vc python=3.10
conda activate starganv2-vc
pip install -r requirements.txt
```


## 推理

```bash
python inference.py \
    --starganv2_vc_model ckpts/starganv2_vc/epoch_0000150.pt \
    --f0_model ckpts/jdc/bst.t7 \
    --hifigan_model ckpts/hifigan/g_00082000 \
    --source_path assets/SSB00120013.wav \
    --ref_path assets/Zh_7_prompt.wav \
    --output_dir outputs/inference/vc \
```


## 数据准备

。。。


## 训练

```bash
python train.py --config_path ./configs/configs_vc.yml
```

Please specify the training and validation data in `config.yml` file. Change `num_domains` to the number of speakers in the dataset. The data list format needs to be `filename.wav|speaker_number`, see [train_list.txt](https://github.com/yl4579/StarGANv2-VC/blob/main/Data/train_list.txt) as an example. 

Checkpoints and Tensorboard logs will be saved at `outputs`. To speed up training, you may want to make `batch_size` as large as your GPU RAM can take. However, please note that `batch_size = 5` will take around 10G GPU RAM. 


## References
- [yl4579/StarGANv2-VC](https://github.com/yl4579/StarGANv2-VC.git)
- [clovaai/stargan-v2](https://github.com/clovaai/stargan-v2)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [tosaka-m/japanese_realtime_tts](https://github.com/tosaka-m/japanese_realtime_tts)
- [keums/melodyExtraction_JDC](https://github.com/keums/melodyExtraction_JDC)
