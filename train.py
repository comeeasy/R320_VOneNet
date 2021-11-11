import torch
from torch._C import device
import torch.optim as optim
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
from torchvision import datasets

import vonenet.vonenet as vonenet
import vonenet.back_ends as back_ends

import utils.data as data
from tqdm import tqdm
import sys
import logging
import time
import config

def model_train(epochs, batch_size, lr, image_size, model_arch, dset_root, dataset, is_vonenet):
    if not model_arch in ["resnet18", "resnet50"] :
        logging.error(f"model_arch: {model_arch}")
        raise ValueError()

    device = config.ConfigTrain.device
    logging.info(f"train with {device}")

    # get data
    logging.info("loading data")
    if dataset.lower() == 'mnist':
        train_data, label_data = data.get_mnist(batch_size, image_size=image_size)
        in_channel = 1
    elif dataset.lower() == 'imagenet':
        train_data, label_data = data.get_imagenet(
            root=dset_root,
            img_size=image_size,
            batch_size=batch_size,
            num_worker=8
        )
        in_channel = 3
    else:
        raise RuntimeError("Not Exist dataset Error")

    logging.info(f"load {model_arch}")
    if config.ConfigTrain.resume:
        start_epoch = config.ConfigTrain.start_epoch
        model = torch.load(config.ConfigTrain.resume_model_path)
    else:
        if is_vonenet: 
            model = vonenet.VOneNet(model_arch=model_arch, in_channel=in_channel)
            model_arch = "VOne" + model_arch
        else:
            model = back_ends.Resnet18(bottleneck_connection_channel=3)

    model = model.train().to(device)

    logging.info(f"train {model_arch} with {dataset}")
    logging.info(f"load {model_arch}")

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logging.info(f"optimizer: {optimizer}")
    logging.info(f"criterion: {criterion}")

    with tensorboard.SummaryWriter() as writer:
        iter = 0
        for epoch in range(start_epoch + 1, start_epoch + epochs + 1):
            total_batch = len(train_data)

            for imgs, targets in tqdm(train_data):
                # for convolutional model
                imgs = imgs.to(device)
                targets = targets.to(device)

                prediction = model(imgs)
                cost = criterion(prediction, targets)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                writer.add_scalar('Loss/train', cost / total_batch, iter)
                iter += 1

            # save weights
            model_path = f"./weights/{model_arch}-{dataset}-ep{epoch:03d}-{time.strftime('%Y-%m-%d-%H')}.pth" 
            torch.save(model, model_path)
            logging.info(f"weight is saved as {model_path}")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    epochs = config.ConfigTrain.epochs
    batch_size = config.ConfigTrain.batch_size
    model_arch = config.ConfigTrain.model_arch
    learning_rate = config.ConfigTrain.learning_rate
    image_size = (config.ConfigTrain.img_size, config.ConfigTrain.img_size)
    dataset = config.ConfigTrain.dataset
    dset_root = config.ConfigTrain.dset_root
    is_vonenet = config.ConfigTrain.is_vonenet
    gpu_device = config.ConfigTrain.device

    logging.info(f"As resnet was trained with ImageNet dataset, image size is fixed as (224, 224)")
    logging.info(f"epochs       : {epochs}")
    logging.info(f"batch size   : {batch_size}")
    logging.info(f"model_arch   : {model_arch}")
    logging.info(f"learning_rate: {learning_rate}")
    logging.info(f"image size   : {image_size}")
    logging.info(f"dataset      : {dataset}")
    logging.info(f"device       : {gpu_device}")

    model_train(epochs, batch_size, learning_rate, image_size, model_arch, dset_root, dataset, is_vonenet) 
