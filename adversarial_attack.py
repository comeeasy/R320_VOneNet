import torch
import data

def generate_image_adversary(model, img_batch, target_batch, eps=0.35, device='cuda'):
    len = img_batch.size(0)
    rand_idx = torch.randperm(len)

    # print(rand_idx)

    img = img_batch.clone().to(device)
    label = target_batch.clone().to(device)

    for i in range(len):
        img[i] = img_batch[rand_idx[i]]
        label[i] = target_batch[rand_idx[i]]

    img = img.view(-1, 28 * 28)
    # print(img.shape)
    # print(label.shape)

    img.requires_grad = True

    # model.zero_grad()
    pred = model(img).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(pred, label).to(device)

    # print(f'loss : {loss}')

    # we need to calculate ∇xJ(x,θ)
    loss.backward()
    img.requires_grad = False

    img = img + eps*img.grad.data.sign()
    # img = torch.clamp(img, 0, 1)

    return img, label

import matplotlib.pyplot as plt
if __name__ == '__main__':
    print('[INFO] getting data...')
    mnist_train, _ = data.get_mnist(batch_size=64)

    print('[INFO] getting model...')
    model = torch.load(f='./weights/trained_MNISTnet.pt')
    model.eval()
    model = model.to('cuda')

    for img_batch, target_batch in mnist_train:
        print('[INFO] getting adversarial imgs...')
        adv_mnist_train, adv_mnist_target = generate_image_adversary(model=model,
                                                   img_batch=img_batch,
                                                   target_batch=target_batch)

        img_batch = img_batch.view(-1, 28, 28)
        # adv_mnist_train.view(-1, 28, 28).to('cpu')
        adv_mnist_train = adv_mnist_train.view(-1, 28, 28).cpu()

        plt.imshow(adv_mnist_train[0])
        print(adv_mnist_target[0])
        plt.show()
        print('///////////////////////////////////////////////////')

        plt.imshow(img_batch[0])
        print(target_batch[0])
        plt.show()

        print('///////////////////////////////////////////////////')

