from tqdm import tqdm
from utils.adversarial_attack import generate_image_adversary
from utils.val_method import Fgsm, DamageNet

import torch
import torch.utils.tensorboard as tensorboard
import torch.optim as optim
import torch.nn as nn

import utils.data as data

import logging
import sys
import os
import config


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def validation(epochs, batch_size, learning_rate, model_path, model_arch, 
                image_size, dataset, dset_root, val_method,
                damagenet_root):

    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} is not exist")

    logging.info(f"validation start with {model_path}")
    logging.info(f"accuracy of original data, adversarial data will be written at tensorboard. please check latest data if runs directory.")
    logging.info(f"Usage: $ tensorboard --logdir ./runs --bind_all")

    logging.info(f"loading dataset {dataset}")
    if dataset.lower() == 'mnist':
        train_dset, val_dset = data.get_mnist(batch_size=batch_size, image_size=image_size)
    elif dataset.lower() == 'imagenet':
        train_dset, val_dset = data.get_imagenet(root=dset_root, img_size=image_size, batch_size=batch_size)

    device = config.ConfigVal.device
    logging.info(f"Use {device}")

    logging.info(f"loading model {model_path}")
    model = torch.load(model_path)
    model.eval().to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    logging.info(f"optimizer: {optimizer}")
    logging.info(f"criterion: {criterion}")

    logging.info("start validation")
    logging.info(f"val_method: {val_method}")

    if val_method.lower() == 'fgsm':
        with tensorboard.SummaryWriter() as writer:
            Fgsm.calc_accuracy(model, val_dset, writer, epoch=0) # first performance
            Fgsm.finetune(model, epochs, train_dset, dataset, criterion, optimizer, model_arch, val_dset, writer)

    elif val_method.lower() == 'damagenet':
        if dataset.lower() != 'imagenet':
            raise RuntimeError("To validate with DAmageNet, must choose imagenet as dataset")
        
        with tensorboard.SummaryWriter() as writer:
            DamageNet.calc_accuracy(model, damagenet_root, dset_root, image_size, batch_size, writer, epoch=0)
            DamageNet.finetune(model, model_arch, epochs, damagenet_root, dset_root, dataset, 
                        image_size, batch_size, criterion, optimizer, writer)


if __name__ == '__main__' :
    
    epochs = config.ConfigVal.epochs_finetune
    batch_size = config.ConfigVal.batch_size_finetune
    learning_rate = config.ConfigVal.learning_rate_finetune
    model_path = config.ConfigVal.model_path
    model_arch = config.ConfigVal.model_arch
    image_size = (config.ConfigVal.img_size, config.ConfigVal.img_size)
    dataset = config.ConfigVal.dataset
    dset_root = config.ConfigVal.dset_root
    val_method = config.ConfigVal.val_method
    damagenet_root = config.ConfigVal.damagenet_root
    resume = config.ConfigVal.resume
    start_epoch = config.ConfigVal.start_epoch

    logging.info(f"As resnet was trained with ImageNet dataset, image size is fixed as (224, 224)")

    logging.info(f"epochs       : {epochs}")
    logging.info(f"batch size   : {batch_size}")
    logging.info(f"learning_rate: {learning_rate}")
    logging.info(f"model_path   : {model_path}")
    logging.info(f"model_arch   : {model_arch}")
    logging.info(f"image size   : {image_size}")
    logging.info(f"dataset      : {dataset}")
    logging.info(f"dset_root    : {dset_root}")
    logging.info(f"resume       : {resume}")
    logging.info(f"start_epoch  : {start_epoch}")
    
    if dataset.lower() == 'damagenet':
        if not os.path.exists(damagenet_root):
            raise FileNotFoundError("DAmageNet path")
        logging.info(f"damagenet_root   : {damagenet_root}")


    logging.info(f"As resnet was trained with ImageNet dataset, image size is fixed as (224, 224)")

    validation(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
               model_path=model_path, model_arch=model_arch, image_size=image_size, 
               dataset=dataset, dset_root=dset_root, val_method=val_method,
               damagenet_root=damagenet_root)
