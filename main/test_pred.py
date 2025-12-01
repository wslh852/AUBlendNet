#!/usr/bin/env python
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import sys
sys.path.append('/home/lh/lihao/AU/Aublendnet-main')
from datetime import datetime
from base.utilities import get_parser, get_logger
from models import get_model
from base.baseTrainer import load_state_dict
from utils.util import *

cfg = get_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.test_gpu)

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def main():
    global cfg, logger
    args = get_parser()
    logger = get_logger()
    logger.info(cfg)
    logger.info("=> creating model ...")
    model = get_model(cfg)
    model = model.cuda()

    if os.path.isfile(cfg.model_path):
        logger.info("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'], strict=False)
        logger.info("=> loaded checkpoint '{}'".format(cfg.model_path))
    else:
        raise RuntimeError("=> no checkpoint flound at '{}'".format(cfg.model_path))

    # ####################### Data Loader ####################### #
    from dataset.data_loader import get_dataloaders
    dataset = get_dataloaders(cfg)
    test_loader = dataset['test']
    #diffusion = create_gaussian_diffusion(args)
    test(args, model, test_loader)

def test(args, model, test_loader):
    model.eval()
    model = model.cuda()
    save_folder = os.path.join(cfg.save_folder, 'npy')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    loss_fn = nn.MSELoss()
    index = 0
    control_base_total = 0
    emotion_total = 0
    with torch.no_grad():
        for i, (data, template, name,onehot) in enumerate(test_loader):
            index = index + 1
           # audio = audio.cuda(non_blocking=True)
            onehot = onehot.cuda(non_blocking=True)
            data = data.cuda(non_blocking=True)
            template = template.cuda(non_blocking=True)         
            prediction = model.predict(template, data, onehot)
            prediction = prediction.squeeze() 
            predicted_auvertices_path = os.path.join(save_folder, 'au.npy')
            #np.save(predicted_auvertices_path, prediction.detach().cpu().numpy())
            loss = loss_fn(prediction,data[0])
            control_base_loss = loss_fn(prediction,data[0])
            control_base_total = control_base_loss + control_base_total

            output_dir = os.path.join(save_folder, str(name[0]))
            if os.path.exists(output_dir) == False:
                os.makedirs(output_dir)
            np.save(os.path.join(output_dir ,str(int(name[0]))+ ".npy"), prediction.detach().cpu().numpy())
            np.save(os.path.join(output_dir ,str((name[0]))+'_gt' ".npy"), data.detach().cpu().numpy())

if __name__ == '__main__':
    main()
