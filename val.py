import torch
import data
from tqdm import tqdm
from adversarial_attack import generate_image_adversary

import vonenet.vonenet as vonenet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def val(attack=False, image_size=224) :
    _, mnist_test = data.get_mnist(image_size=image_size)

    # model = torch.load(f='./weights/tmp.pt')
    # model.eval()
    # model = model.to(device)

    # model_path = './weights/original-data/AlexNet-epochs-10-epochs-10.pt'
    # model_path = './weights/original-data/ConvNet-epochs-10-epochs-10.pt'
    # model_path = './weights/original-data/Basic_CNN-epochs-10.pt'
    model_path = './weights/original-data/Basic_Linear_Regression-epochs-10.pt'
    model_name = 'Basic-CNN'

    model = torch.load(model_path)
    model.eval()
    model = model.to(device)

    ####################################################
    print(f'\n[INFO] model        : {model_path}')
    print(f'[INFO] image size   : {image_size}')
    ####################################################

    if attack == False :
        print(f'\n[INFO] calculate accuracy of original image for original model of {model_name}')
        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in tqdm(mnist_test):
            with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                # for linear model
                X_test = img_batch.view(-1, 28 * 28).to(device)

                # for convolutional model
                # X_test = img_batch.to(device)
                Y_test = target_batch.to(device)

                prediction = model(X_test).to(device)
                correct_prediction = torch.argmax(prediction, 1) == Y_test
                accuracy = correct_prediction.float().mean()

                accuracy_avg += accuracy / total_batch

        print(f'[RESULT] accuracy of original image for original model of {model_name} == {accuracy_avg * 100:5f}%')
        return accuracy_avg

    else :
        print(f'\n[INFO] calculate accuracy of adversarial image for original model of {model_name}')

        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in tqdm(mnist_test) :
            # print(mnist_test_batch[0].shape)
            X_test, Y_test = generate_image_adversary(model=model, img_batch=img_batch,target_batch=target_batch)

            with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                # for linear model
                X_test = X_test.view(-1, 28 * 28).to(device)
                # for convolutional model
                # X_test = X_test.to(device)
                Y_test = Y_test.to(device)

                prediction = model(X_test).to(device)
                correct_prediction = torch.argmax(prediction, 1) == Y_test
                accuracy = correct_prediction.float().mean()

                accuracy_avg += accuracy / total_batch

        print(f'[RESULT] accuracy of adversarial image for original model of {model_name} == {accuracy_avg * 100:5f}%')
        return accuracy_avg

def fine_tuned_val(model_path, model_name, attack=False, image_size=224) :
    _, mnist_test = data.get_mnist(image_size=image_size)

    model = torch.load(f=model_path)
    model.eval()
    model = model.to(device)

    ####################################################
    # print(f'\n[INFO] model      : {model_path}')
    # print(f'[INFO] model name   : {model_name}')
    # print(f'[INFO] image size   : {image_size}')
    ####################################################

    if attack == False :
        # print(f'\n[INFO] calculate accuracy of original image for fine-tuned model of {model_name}')

        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in mnist_test:
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

        # print(f'[RESULT] accuracy of original image for fine-tuned model of {model_name} == {accuracy_avg * 100:5f}%')
        return accuracy_avg

    else :
        # print(f'\n[INFO] calculate accuracy of adversarial image for fine-tuned model of {model_name}')

        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in mnist_test :
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

        # print(f'[RESULT] accuracy of adversarial image for fine-tuned model of {model_name} == {accuracy_avg * 100:5f}%')
        return accuracy_avg

import train
import matplotlib.pyplot as plt
import numpy as np
import os

def validation(epochs, pre_trained, image_size):
    model_path = './weights/original-data/AlexNet-epochs-10.pt'
    model_name = 'AlexNet'
    tmp_path = './weights/tmp.pt'

    print(f'[INFO] validation start')
    print(f'[INFO] epochs:{epochs}, pre_trained:{pre_trained}, img_size:{image_size}')
    print(f'[INFO] model name : {model_name}')

    print(f'[INFO] saving pre-trained model to {tmp_path}')
    os.system(f'cp {model_path} {tmp_path}')
    print(f'cp {model_path} {tmp_path}')



    ori_acc_list = []
    adv_acc_list = []

    if not pre_trained:
        print(f'[INFO] pre-train because model has not been trained')
        train.model_train(epochs=10, image_size=image_size)

    print(f'[INFO] fine-tune {model_name} via tmp.pt')
    for epoch in tqdm(range(1, epochs + 1)):
        print(f'[INFO] epochs : {epoch}')

        ori_acc = fine_tuned_val(model_path=tmp_path, model_name=model_name, attack=False, image_size=image_size) * 100
        adv_acc = fine_tuned_val(model_path=tmp_path, model_name=model_name, attack=True, image_size=image_size) * 100

        ori_acc_list.append(ori_acc)
        adv_acc_list.append(adv_acc)

        print('=======================================================================')
        print(f'fine-tuned model original    image accuracy : {ori_acc:5f}%')
        print(f'fine-tuned model adversarial image accuracy : {adv_acc:5f}%\n')
        print('=======================================================================')

        if epoch == epochs: break
        train.fine_tune(epochs=1, image_size=image_size)

    epochs_range = range(epochs)

    # red dashes, blue squares and green triangles
    plt.plot(epochs_range, ori_acc_list, 'r.-', label='origin      image accuracy')
    plt.plot(epochs_range, adv_acc_list, 'b.-', label='adversarial image accuracy')

    try:
        plt.legend(loc='lower right')
    except :
        print(f'Exception')

    plt.title(model_name)
    plt.axis([0, epochs, 0, 100])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.yticks(np.arange(start=0, stop=101, step=10))
    plt.grid(True, axis='y')
    plt.savefig('./final-report/fine-tuned/' + model_name + '.jpg')

    print(f'[INFO] graph saved in path ./fine-tuned/{model_name}.jpg')

    # plt.show()




if __name__ == '__main__' :
    # train.model_train(epochs=5)
    # train.fine_tune(epochs=2)

    # val(attack=False, image_size=28)
    # val(attack=True, image_size=28)
    # print(f'fine-tuned model accuracy : {fine_tuned_val(attack=False) * 100}')
    # print(f'fine-tuned model accuracy : {fine_tuned_val(attack=True) * 100}')

    validation(epochs=30, pre_trained=True, image_size=28)

