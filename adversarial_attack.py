import torch
import data

def generate_image_adversary(model, img_batch, target_batch, eps=0.35, device='cuda'):
    global i

    # len = img_batch.size(0)
    # rand_idx = torch.randperm(len)

    # print(rand_idx)

    img = img_batch.clone().to(device)
    label = target_batch.clone().to(device)

    # for i in range(len):
    #     img[i] = img_batch[rand_idx[i]]
    #     label[i] = target_batch[rand_idx[i]]

    # for linear model
    # img = img.view(-1, 28 * 28)

    img.requires_grad = True

    # model.zero_grad()
    pred = model(img).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(pred, label).to(device)

    # we need to calculate ∇xJ(x,θ)
    # with torch.autograd.set_detect_anomaly(True):
    loss.backward()

    img.requires_grad = False

    img = img + eps*img.grad.data.sign()
    # img = torch.clamp(img, 0, 1)

    return img, label

import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    image_size = 28
    device = 'cuda'
    batch_size = 64
    _, mnist_val_dateloader = data.get_mnist(batch_size, image_size=image_size)

    model_weights = './weights/original/ConvNet-epochs-10.pt'

    # print(f'[INFO] getting model {model_weights}')
    model = torch.load(f=model_weights)
    model.eval()
    model = model.to(device)
    i = 0
    for img_batch, target_batch in tqdm(mnist_val_dateloader):
        ori_sample = img_batch.__getitem__(0)
        ori_sample = ori_sample.to('cpu')
        ori_sample = torch.reshape(ori_sample, [28, 28]).numpy()
        plt.imsave(f'./final-report/adv-img-samples/ori-sample{i}.bmp', ori_sample)

        # generate adversarial image batch
        adv_img_batch, adv_target_batch = generate_image_adversary(model=model, img_batch=img_batch,
                                                                   target_batch=target_batch)
        adv_img_batch = adv_img_batch.to(device)
        adv_target_batch = adv_target_batch.to(device)

        adv_sample = adv_img_batch.__getitem__(0)
        adv_sample = adv_sample.to('cpu')
        adv_sample = torch.reshape(adv_sample, [28, 28]).numpy()
        plt.imsave(f'./final-report/adv-img-samples/adv-sample{i}.bmp', adv_sample)
        i += 1



