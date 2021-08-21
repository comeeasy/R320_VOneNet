import torch
import torch.optim as optim
import torch.nn as nn

import data
from tqdm import tqdm

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

        for img_batch, target_batch in tqdm(train_data):
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