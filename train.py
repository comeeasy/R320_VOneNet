import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
from torchvision import datasets

import vonenet.vonenet as vonenet
import vonenet.back_ends as back_ends

import data
from tqdm import tqdm
import sys
import logging
import argparse



def vonenet_model_train(epochs, batch_size, lr, image_size, model_arch, dataset):
    if not model_arch in ["resnet18", "resnet50"] :
        logging.error(f"model_arch: {model_arch}")
        raise ValueError()

    logging.info(f"train vonenet-{model_arch} with MNIST data")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"train with {device}")

    # get data
    logging.info("loading data")
    if dataset.lower() == 'mnist':
        train_data, label_data = data.get_mnist(batch_size, image_size=image_size)
        in_channel = 1
    elif dataset.lower() == 'imagenet':
        train_data, label_data = data.get_imagenet(
            root='/media/r320/2d365830-836f-4d91-8998-fef7c8443335/ImageNet_dset/ILSVRC2012',
            img_size=image_size,
            batch_size=batch_size,
            num_worker=8
        )
        in_channel = 3
    else:
        raise RuntimeError("Not Exist dataset")

    logging.info(f"load vonenet-{model_arch}")
    model = vonenet.VOneNet(model_arch=model_arch, in_channel=in_channel)
    model = model.train().to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logging.info(f"optimizer: {optimizer}")
    logging.info(f"criterion: {criterion}")

    with tensorboard.SummaryWriter() as writer:
        iter = 0
        for epoch in range(epochs):
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
            torch.save(model, f"./weights/{model_arch}-ep{epoch:3d}.pth")
            print(f"./weights/{model_arch}-ep{epoch:03d}.pth")


def model_train(epochs, batch_size, lr, image_size, model_arch, dataset):
    if not model_arch in ["resnet18", "resnet50"] :
        logging.error(f"model_arch: {model_arch}")
        raise ValueError()

    logging.info(f"train {model_arch} with {dataset}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"train with {device}")

    # get data
    logging.info("loading data")
    if dataset.lower() == 'mnist':
        train_data, label_data = data.get_mnist(batch_size, image_size=image_size)
        in_channel = 1
    elif dataset.lower() == 'imagenet':
        train_data, label_data = data.get_imagenet(
            root='/media/r320/2d365830-836f-4d91-8998-fef7c8443335/ImageNet_dset/ILSVRC2012',
            img_size=image_size,
            batch_size=batch_size,
            num_worker=8
        )
        in_channel = 3
    else:
        raise RuntimeError("Not Exist dataset")

    logging.info(f"load {model_arch}")
    model = back_ends.Resnet18(bottleneck_connection_channel=3)
    model = model.train().to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logging.info(f"optimizer: {optimizer}")
    logging.info(f"criterion: {criterion}")

    with tensorboard.SummaryWriter() as writer:
        iter = 0
        for epoch in range(epochs):
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
            torch.save(model, f"./weights/{model_arch}-{dataset}-ep{epoch:3d}.pth")
            print(f"./weights/{model_arch}-ep{epoch:03d}.pth")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VOneNet training Usage')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_arch', type=str, required=True, default="resnet18",
                        help='arch: ["resnet18", "resnet18_28x28"] default="resnet18"')
    parser.add_argument('--img_size', type=int, required=True, default=224,
                        help='shape=(img_size, img_size')
    parser.add_argument('--dataset', type=str, required=True, default='imagenet',
                        help='choose one of [imagenet , mnist]')

    FLAGS, FIRE_FLAGS = parser.parse_known_args()

    epochs = int(FLAGS.epochs)
    batch_size = int(FLAGS.batch_size)
    model_arch = FLAGS.model_arch
    learning_rate = float(FLAGS.lr)
    image_size = (int(FLAGS.img_size), int(FLAGS.img_size))
    dataset = FLAGS.dataset

    logging.info(f"As resnet was trained with ImageNet dataset, image size is fixed as (224, 224)")
    logging.info(f"epochs       : {epochs}")
    logging.info(f"batch size   : {batch_size}")
    logging.info(f"model_arch   : {model_arch}")
    logging.info(f"learning_rate: {learning_rate}")
    logging.info(f"image size   : {image_size}")
    logging.info(f"dataset      : {dataset}")

    #vonenet_model_train(epochs=epochs, batch_size=batch_size, lr=learning_rate,
    #                    image_size=image_size, model_arch=model_arch, dataset=dataset)

    model_train(epochs, batch_size, learning_rate, image_size, model_arch, dataset) 
