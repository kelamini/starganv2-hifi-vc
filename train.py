#!/usr/bin/env python3
#coding:utf-8

import os.path as osp

from munch import Munch
import torch
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter('ignore')

from src.data.meldataset import build_dataloader
from src.models.optimizers import build_optimizer
from src.models.vc.models import build_model
from src.models.asr.models import ASRCNN
from src.models.jdc.model import JDCNet
from src.handler.trainer import Trainer
from src.configs import config

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True #


def main(config):

    log_dir = config['log_dir']
    writer = SummaryWriter(log_dir + "/logs")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)
    
    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)
    stage = config.get('stage', 'star')
    fp16_run = config.get('fp16_run', False)
    
    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=4,
                                        device=device)
    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device)

    # load pretrained ASR model
    asr_model_path = config['asr_params']['model_path']
    asr_model_config = config['asr_params']['model_params']
    asr_model = ASRCNN(**asr_model_config)
    params = torch.load(asr_model_path, map_location='cpu')['model']
    asr_model.load_state_dict(params)
    _ = asr_model.eval()
    
    # load pretrained F0 model
    f0_model_path = config['f0_params']['model_path']
    f0_model_config = config['f0_params']['model_params']
    f0_model = JDCNet(**f0_model_config)
    params = torch.load(f0_model_path, map_location='cpu')['net']
    f0_model.load_state_dict(params)
    
    # build model
    model, model_ema = build_model(Munch(config['model_params']), f0_model, asr_model)

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 2e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    
    _ = [model[key].to(device) for key in model]
    _ = [model_ema[key].to(device) for key in model_ema]
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['mapping_network']['max_lr'] = 2e-6
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                      scheduler_params_dict=scheduler_params_dict)

    trainer = Trainer(args=Munch(config['loss_params']), model=model,
                            model_ema=model_ema,
                            optimizer=optimizer,
                            device=device,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            logger=logger,
                            fp16_run=fp16_run,
                            writer=writer)

    if config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(config['pretrained_model'],
                                load_only_params=config.get('load_only_params', True))

    for _ in range(1, epochs+1):
        epoch = trainer.epochs
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info('%-15s: %.4f' % (key, value))
                # writer.add_scalar(key, value, epoch)
            # else:
            #     for k, v in value[0].items():
                    # writer.add_figure(f'eval_spec/{k}', v, epoch*len(val_dataloader))
        if (epoch % save_freq) == 0:
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))

    return 0


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8') as f:
        val_list = f.readlines()

    return train_list, val_list



if __name__=="__main__":
    main(config)
