import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.tensorboard as tensorboard

import vonenet.vonenet as vonenet

import data
from tqdm import tqdm
import sys
import logging
import argparse



def vonenet_model_train(epochs, batch_size, lr, image_size, model_arch):
    if model_arch != "resnet18":
        logging.error(f"model_arch: {model_arch}")
        raise ValueError()

    logging.info(f"train vonenet-{model_arch} with MNIST data")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"train with {device}")

    # get data
    logging.info("loading data")
    train_data, label_data = data.get_mnist(batch_size, image_size=image_size)

    logging.info(f"load vonenet-{model_arch}")
    model = vonenet.VOneNet(model_arch=model_arch)
    model.eval().to(device)

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
            print(f"./weights/{model_arch}-ep{epoch:3d}.pth")




logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VOneNet training Usage')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_arch', type=str, required=True, default="resnet18",
                        help='arch: ["resnet18"]')

    FLAGS, FIRE_FLAGS = parser.parse_known_args()

    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    model_arch = FLAGS.model_arch
    learning_rate = FLAGS.lr

    logging.info(f"As resnet was trained with ImageNet dataset, image size is fixed as (224, 224)")
    image_size = (224, 224)

    logging.info(f"epochs       : {epochs}")
    logging.info(f"batch size   : {batch_size}")
    logging.info(f"model_arch   : {model_arch}")
    logging.info(f"learning_rate: {learning_rate}")
    logging.info(f"image size    : {image_size}")

    vonenet_model_train(epochs=epochs, batch_size=batch_size, lr=learning_rate,
                        image_size=image_size, model_arch=model_arch)
