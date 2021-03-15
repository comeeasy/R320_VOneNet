import torch
import data
from tqdm import tqdm
from adversarial_attack import generate_image_adversary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def val(attack=False) :
    _, mnist_test = data.get_mnist()

    model = torch.load(f='./weights/trained_MNISTnet.pt')
    model.eval()
    model = model.to(device)

    if attack == False :
        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in tqdm(mnist_test):
            with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                X_test = img_batch.view(-1, 28 * 28).to(device)
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
            X_test = generate_image_adversary(model=model, img_batch=img_batch,target_batch=target_batch)

            with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                X_test = X_test.view(-1, 28 * 28).to(device)
                Y_test = target_batch.to(device)

                prediction = model(X_test).to(device)
                correct_prediction = torch.argmax(prediction, 1) == Y_test
                accuracy = correct_prediction.float().mean()

                accuracy_avg += accuracy / total_batch

        return accuracy_avg

def fine_tuned_val(attack=False) :
    _, mnist_test = data.get_mnist()

    model = torch.load(f='./weights/fine-tuned_trained_MNISTnet.pt')
    model.eval()
    model = model.to(device)

    if attack == False :
        accuracy_avg = 0
        total_batch = len(mnist_test)

        for img_batch, target_batch in tqdm(mnist_test):
            with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                X_test = img_batch.view(-1, 28 * 28).to(device)
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
            X_test = generate_image_adversary(model=model, img_batch=img_batch,target_batch=target_batch)

            with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
                X_test = X_test.to(device)
                Y_test = target_batch.to(device)

                prediction = model(X_test).to(device)
                correct_prediction = torch.argmax(prediction, 1) == Y_test
                accuracy = correct_prediction.float().mean()

                accuracy_avg += accuracy / total_batch

        return accuracy_avg

if __name__ == '__main__' :
    print(f'before adversarial attack, accuracy : {val(attack=False) * 100}')
    print(f'after  adversarial attack, accuracy : {val(attack=True) * 100}')
    print(f'fine-tuned model accuracy : {fine_tuned_val(attack=False) * 100}')
    print(f'fine-tuned model accuracy : {fine_tuned_val(attack=True) * 100}')


