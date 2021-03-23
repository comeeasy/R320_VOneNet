import torch
import data
from tqdm import tqdm
from adversarial_attack import generate_image_adversary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def val(attack=False) :
    _, mnist_test = data.get_mnist()

    model = torch.load(f='./weights/tmp.pt')
    model.eval()
    model = model.to(device)

    if attack == False :
        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in tqdm(mnist_test):
            with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                # for linear model
                # X_test = img_batch.view(-1, 28 * 28).to(device)

                # for convolutional model
                X_test = img_batch.to(device)
                Y_test = target_batch.to(device)

                prediction = model(X_test).to(device)
                correct_prediction = torch.argmax(prediction, 1) == Y_test
                accuracy = correct_prediction.float().mean()

                accuracy_avg += accuracy / total_batch

        return accuracy_avg

    else :
        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in tqdm(mnist_test) :
            # print(mnist_test_batch[0].shape)
            X_test, Y_test = generate_image_adversary(model=model, img_batch=img_batch,target_batch=target_batch)

            with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                # for linear model
                # X_test = X_test.view(-1, 28 * 28).to(device)
                # for convolutional model
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)

                prediction = model(X_test).to(device)
                correct_prediction = torch.argmax(prediction, 1) == Y_test
                accuracy = correct_prediction.float().mean()

                accuracy_avg += accuracy / total_batch

        return accuracy_avg

def fine_tuned_val(attack=False) :
    _, mnist_test = data.get_mnist()

    model = torch.load(f='./weights/tmp.pt')
    model.eval()
    model = model.to(device)

    if attack == False :
        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in tqdm(mnist_test):
            with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                # for linear model
                # X_test = img_batch.view(-1, 28 * 28).to(device)
                # for convolutional model
                X_test = img_batch.to(device)
                Y_test = target_batch.to(device)

                prediction = model(X_test).to(device)
                correct_prediction = torch.argmax(prediction, 1) == Y_test
                accuracy = correct_prediction.float().mean()

                accuracy_avg += accuracy / total_batch

        return accuracy_avg

    else :
        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in tqdm(mnist_test) :
            X_test, Y_test = generate_image_adversary(model=model, img_batch=img_batch,target_batch=target_batch)

            with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                # for linear model
                # X_test = X_test.view(-1, 28 * 28).to(device)
                # for convolutional model
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)

                prediction = model(X_test).to(device)
                correct_prediction = torch.argmax(prediction, 1) == Y_test
                accuracy = correct_prediction.float().mean()

                accuracy_avg += accuracy / total_batch

        return accuracy_avg

import train
import matplotlib.pyplot as plt
import numpy as np

def validation(epochs=100):
    file = "./report2/Alexnet-tanh-tanh.log"
    title = 'Alexnet-tanh-tanh'
    f = open(file=file, mode='w', encoding='utf-8')

    f.write(title +'\n')

    ori_acc_list = []
    adv_acc_list = []

    train.model_train(epochs=3)
    print(f'[INFO] epochs : {epochs}')

    for epoch in range(1, epochs + 1):
        print(f'[INFO] epochs : {epoch}')

        f.write(f'\nepoch : {epoch}\n')

        ori_acc = fine_tuned_val(attack=False) * 100
        adv_acc = fine_tuned_val(attack=True) * 100

        ori_acc_list.append(ori_acc)
        adv_acc_list.append(adv_acc)

        f.write(f'fine-tuned model original    image accuracy : {ori_acc:3f}%\n')
        f.write(f'fine-tuned model adversarial image accuracy : {adv_acc:3f}%\n')

        train.fine_tune(epochs=1)

    epochs_range = range(epochs)

    # red dashes, blue squares and green triangles
    plt.plot(epochs_range, ori_acc_list, 'r.-', label='origin      image accuracy')
    plt.plot(epochs_range, adv_acc_list, 'b.-', label='adversarial image accuracy')

    plt.title(title)
    plt.axis([0, epochs, 0, 100])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.yticks(np.arange(start=0, stop=101, step=10))
    plt.grid(True, axis='y')
    plt.show()




if __name__ == '__main__' :
    train.model_train(epochs=5)
    # train.fine_tune(epochs=2)

    print(f'before adversarial attack, accuracy : {val(attack=False) * 100}')
    print(f'after  adversarial attack, accuracy : {val(attack=True) * 100}')
    print(f'fine-tuned model accuracy : {fine_tuned_val(attack=False) * 100}')
    print(f'fine-tuned model accuracy : {fine_tuned_val(attack=True) * 100}')

    # validation(epochs=30)

