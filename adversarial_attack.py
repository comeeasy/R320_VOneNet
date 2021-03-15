import torch
import data

def generate_image_adversary(model, img_batch, target_batch, eps=0.35, device='cuda'):
    img = img_batch.float().to(device)
    label = target_batch.to(device)

    len = img.size(0)
    rand_idx = torch.randperm(len)
    tmp = img
    tmp_l = label

    for i in range(len):
        img[i] = tmp[rand_idx[i]]
        label[i] = tmp_l[rand_idx[i]]
    
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

    tmp = img + eps*img.grad.data.sign()
    tmp = torch.clamp(tmp, 0, 1)

    return tmp

import matplotlib.pyplot as plt
if __name__ == '__main__':
    print('[INFO] getting data...')
    mnist_train, _ = data.get_mnist()

    print('[INFO] getting model...')
    model = torch.load(f='./weights/trained_Alexnet.pt')
    model.eval()
    model = model.to('cuda')

    _, img_batch, target_batch = mnist_train
    print('[INFO] getting adversarial imgs...')
    adv_mnist_train = generate_image_adversary(model=model,
                                               img_batch=img_batch,
                                               target_batch=target_batch)
    plt.imshow(img_batch[0])
    plt.show()
    plt.imshow(adv_mnist_train[0])
    plt.show()
