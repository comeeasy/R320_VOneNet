import torch
import torch.utils.tensorboard as tensorboard
import torch.optim as optim
import torch.nn as nn

from adversarial_attack import generate_image_adversary

import data
import logging
import sys
import os
from tqdm import tqdm
import argparse


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def accuracy(attacked, mnist_test, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    accuracy_avg = 0
    total_batch = len(mnist_test)

    for img_batch, target_batch in tqdm(mnist_test):

        if attacked:
            img_batch, target_batch = generate_image_adversary(model=model, img_batch=img_batch, target_batch=target_batch)

        with torch.no_grad():
            X_test = img_batch.to(device)
            Y_test = target_batch.to(device)

            prediction = model(X_test).to(device)
            correct_prediction = torch.argmax(prediction, 1) == Y_test
            accuracy = correct_prediction.float().mean()
            accuracy_avg += accuracy / total_batch

    return accuracy_avg


def validation(epochs, batch_size, learning_rate, model_path, model_arch, image_size):

    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} is not exist")

    logging.info(f"validation start with {model_path}")
    logging.info(f"accuracy of original data, adversarial data will be written at tensorboard. \
                        please check latest data if runs directory")
    logging.info(f"Usage: $ tensorboard --logdir ./runs --bind_all")

    logging.info(f"loading dataset")
    mnist_train, mnist_test = data.get_mnist(batch_size=batch_size, image_size=image_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Use {device}")

    logging.info(f"loading model {model_path}")
    model = torch.load(model_path)
    model.eval().to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    logging.info(f"optimizer: {optimizer}")
    logging.info(f"criterion: {criterion}")

    with tensorboard.SummaryWriter() as writer:
        iter = 0
        for epoch in range(epochs):
            total_batch = len(mnist_train)

            for imgs, targets in tqdm(mnist_train):
                img_batch, target_batch = generate_image_adversary(model=model, img_batch=imgs,
                                                                   target_batch=targets)
                prediction = model(img_batch)
                cost = criterion(prediction, target_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                writer.add_scalar('Loss/train', cost / total_batch, iter)
                writer.add_scalar("Test/origin_accuracy", accuracy(attacked=False, mnist_test=mnist_test, model=model))
                writer.add_scalar("Test/advers_accuracy", accuracy(attacked=True, mnist_test=mnist_test, model=model))
                iter += 1

            # save weights
            torch.save(model, f"./weights/{model_arch}-ep{epoch:3d}-finetuned.pth")
            print(f"./weights/{model_arch}-ep{epoch:3d}-finetuned.pth")

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='VOneNet fine tunning validation Usage')
    parser.add_argument('--epochs', type=int, default=1,
                        help='how many epochs needed for fine tunning validation')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to originally trained vonenet model')
    parser.add_argument('--model_arch', type=str, required=True, default="resnet18",
                        help='arch: ["resnet18"]')

    FLAGS, FIRE_FLAGS = parser.parse_known_args()

    logging.info(f"As resnet was trained with ImageNet dataset, image size is fixed as (224, 224)")
    image_size = (224, 224)

    validation(epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, learning_rate=FLAGS.lr,
               model_path=FLAGS.model_path, model_arch=FLAGS.model_arch)


