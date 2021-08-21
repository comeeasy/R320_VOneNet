import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.tensorboard as tensorboard

import data
import ML_model
from tqdm import tqdm

import vonenet.vonenet as vonenet
import vonenet.back_ends as back_ends

def vonenet_model_train(epochs = 5, batch_size=8, lr=1e-3, image_size=(224, 224), model_arch: str=""):
    if model_arch == "":
        raise ValueError("model arch: resnet18")

    print('[INFO] train with origin data')
    print(f'[INFO] model name : {model_arch}')


    print(f'[INFO] batch_size : {batch_size}, lr : {lr}, img_size: {image_size}')
    print(f'[INFO] epochs : {epochs}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'train with {device}')

    # get data
    train_data, label_data = data.get_mnist(batch_size, image_size=image_size)

    # get model (AlexNet)
    model = vonenet.VOneNet(model_arch=model_arch)
    model.eval()
    model = model.to(device)

    print(f'optimizer       : Adam')
    print(f'cost function   : CrossEntropyLoss')
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with tensorboard.SummaryWriter() as writer:
        for epoch in range(epochs):
            avg_cost = 0
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

                avg_cost += cost / total_batch

            writer.add_scalar('Loss/train', avg_cost, epoch)

            # save weights
            torch.save(model, f"./weights/{model_arch}-ep{epoch:3d}.pth")
            print(f"./weights/{model_arch}-ep{epoch:3d}.pth")

from adversarial_attack import generate_image_adversary

def fine_tune(epochs = 10, image_size=224):
    print('[INFO] fine-tuning with adversarial images')

    # hyperparameter
    batch_size = 64
    learning_rate = 1e-4
    print(f'[INFO] batch_size : {batch_size}, lr : {learning_rate}, img_size: {image_size}')
    # print(f'[INFO] epochs : {epochs}')


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[INFO] train with {device}')

    # get data
    # print('[INFO] getting data')
    train_data, _ = data.get_mnist(batch_size, image_size=image_size)

    # get model

    model_weights = './weights/tmp.pt'

    # print(f'[INFO] getting model {model_weights}')
    model = torch.load(f=model_weights)
    model.eval()
    model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    costF = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        avg_cost = 0
        total_batch = len(train_data)

        for img_batch, target_batch in train_data:
            # generate adversarial image batch
            adv_img_batch, adv_target_batch = generate_image_adversary(model=model, img_batch=img_batch, target_batch=target_batch)

            # for linear model
            # adv_img_batch = adv_img_batch.view(-1, 28 * 28).to(device)
            # for convolutional model
            adv_img_batch = adv_img_batch.to(device)
            adv_target_batch = adv_target_batch.to(device)

            prediction = model(adv_img_batch)
            cost = costF(prediction, adv_target_batch)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        # print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    # save weights
    print('[INFO] saving model in path, ./weights/tmp.pt')
    torch.save(model, 'weights/tmp.pt')

if __name__ == '__main__':
    vonenet_model_train(epochs=10, image_size=(224, 224), model_arch='resnet18')
    # fine_tune()