import torch
import utils.data

import matplotlib.pyplot as plt

from tqdm import tqdm
import os

def generate_image_adversary(model, img_batch, target_batch, eps=0.01, device='cuda'):
    img = img_batch.clone().to(device)
    label = target_batch.clone().to(device)

    img.requires_grad = True

    pred = model(img).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(pred, label).to(device)

    # we need to calculate ∇xJ(x,θ)
    # with torch.autograd.set_detect_anomaly(True):
    loss.backward()
    img.requires_grad = False
    img = img + eps * img.grad.data.sign()

    return img, label




if __name__ == "__main__":

    image_size = int(input("enter image size -> image size: (img_size x img_size)"))
    batch_size = int(input("enter batch size"))
    mnist_data_path = input("enter MNIST data set directory path (default: './MNIST_data')")
    model_path = input("enter model path (*.pth)")

    # convert to absolute path
    mnist_data_path = os.path.abspath(mnist_data_path)
    model_path = os.path.abspath(model_path)
    dst_path = mnist_data_path + '_attacked'
    dst_origin_path = dst_path + "/origin"
    dst_adv_path = dst_path + "/adversarial"

    if not os.path.isdir(mnist_data_path):
        raise NotADirectoryError

    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)
        print(f"{dst_path} has been created")

    if not os.path.isdir(dst_origin_path):
        os.makedirs(dst_origin_path)
        print(f"{dst_origin_path} has been created")

    if not os.path.isdir(dst_adv_path):
        os.makedirs(dst_adv_path)
        print(f"{dst_adv_path} has been created")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mnist_train_dataloader, mnist_val_dataloader = data.get_mnist(batch_size, image_size=image_size)

    # print(f'[INFO] getting model {model_weights}')
    model = torch.load(f=model_path)
    model.eval().to(device)

    i = 0
    for img_batch, target_batch in tqdm(mnist_val_dataloader):
        ori_sample = img_batch.__getitem__(0)
        ori_sample = ori_sample.to('cpu')
        ori_sample = torch.reshape(ori_sample, [image_size, image_size]).numpy()
        # plt.imsave(f'./final-report/VOne-adv-img-samples/V1-ori-sample{i}.bmp', ori_sample)

        # generate adversarial image batch
        adv_img_batch, adv_target_batch = generate_image_adversary(model=model, img_batch=img_batch,
                                                                   target_batch=target_batch)
        adv_img_batch = adv_img_batch.to(device)
        adv_target_batch = adv_target_batch.to(device)

        adv_sample = adv_img_batch.__getitem__(0)
        adv_sample = adv_sample.to('cpu')
        adv_sample = torch.reshape(adv_sample, [28, 28]).numpy()
        # plt.imsave(f'./final-report/VOne-adv-img-samples/V1-adv-sample{i}.bmp', adv_sample)
        i += 1