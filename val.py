import torch
import torch.utils.tensorboard as tensorboard
import torch.optim as optim
import torch.nn as nn

from adversarial_attack import generate_image_adversary
import data

import data
import logging
import sys
import os
from tqdm import tqdm
import argparse
import time


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def accuracy(attacked, val_dset, model, test_size_limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    accuracy_avg = 0
    total_batch = len(val_dset) if not test_size_limit else test_size_limit

    batch_cnt = 0
    for img_batch, target_batch in val_dset:
        if attacked:
            img_batch, target_batch = generate_image_adversary(model=model, img_batch=img_batch, target_batch=target_batch)

        with torch.no_grad():
            X_test = img_batch.to(device)
            Y_test = target_batch.to(device)

            prediction = model(X_test).to(device)
            correct_prediction = torch.argmax(prediction, 1) == Y_test
            accuracy = correct_prediction.float().mean()
            accuracy_avg += accuracy / total_batch

        # for reduce time
        if test_size_limit:
            batch_cnt += 1
            if batch_cnt == test_size_limit:
                break

    return accuracy_avg

def accuracy(img_batch, target_batch, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        X_test = img_batch.to(device)
        Y_test = target_batch.to(device)

        prediction = model(X_test).to(device)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()

    return accuracy


def validation(epochs, batch_size, learning_rate, model_path, model_arch, 
                image_size, test_size_limit, dataset, dset_root, val_method,
                damagenet_root):

    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} is not exist")

    logging.info(f"validation start with {model_path}")
    logging.info(f"accuracy of original data, adversarial data will be written at tensorboard. please check latest data if runs directory.")
    logging.info(f"Usage: $ tensorboard --logdir ./runs --bind_all")

    logging.info(f"loading dataset {dataset}")
    if dataset.lower() == 'mnist':
        train_dset, val_dset = data.get_mnist(batch_size=batch_size, image_size=image_size)
    elif dataset.lower() == 'imagenet':
        train_dset, val_dset = data.get_imagenet(root=dset_root, img_size=image_size, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Use {device}")

    logging.info(f"loading model {model_path}")
    model = torch.load(model_path)
    model.eval().to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    logging.info(f"optimizer: {optimizer}")
    logging.info(f"criterion: {criterion}")

    logging.info("start validation")
    logging.info(f"val_method: {val_method}")

    with tensorboard.SummaryWriter() as writer:
        iter = 0
# ======= FGSM =======================================================================
        if val_method.lower() == 'fgsm':
            for epoch in range(epochs):
                total_batch = len(train_dset)

                for img_batch, target_batch in tqdm(train_dset):
                    writer.add_images("Images/original image batch", img_batch, iter)

                    if dataset.lower() == 'imagenet':
                        adv_img_batch, adv_target_batch = generate_image_adversary(
                            model=model, 
                            img_batch=img_batch,
                            target_batch=target_batch)
                    writer.add_images("Images/adversarial image batch", img_batch, iter)

                    prediction = model(adv_img_batch)
                    cost = criterion(prediction, adv_target_batch)

                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()

                    writer.add_scalar('Loss/train', cost / total_batch, iter)
                    writer.add_scalar("Test/origin_accuracy", accuracy(
                        img_batch=img_batch, 
                        target_batch=target_batch, 
                        model=model), iter)
                    writer.add_scalar("Test/advers_accuracy", accuracy(
                        img_batch=adv_img_batch, 
                        target_batch=adv_target_batch, 
                        model=model), iter)
                    iter += 1

                # save weights
                torch.save(model, f"./weights/{model_arch}-{dataset}-ep{epoch:03d}-FGSM.pth")
                print(f"./weights/{model_arch}-{dataset}-ep{epoch:03d}-FGSM.pth")
# ======= FGSM =======================================================================
# ======= DAmegeNet ==================================================================
        elif val_method.lower() == 'damagenet':
            if dataset.lower() != 'imagenet':
                raise RuntimeError("To validate with DAmageNet, must choose imagenet as dataset")

            damagenet_val = data.get_damegenet(root=damagenet_root, img_size=image_size, batch_size=batch_size)

            for epoch in range(epochs):
                total_batch = len(damagenet_val)

                for img_batch, target_batch in tqdm(damagenet_val):
                    val_dset.next

                    writer.add_images("Images/adversarial image batch", img_batch, iter)

                    prediction = model(img_batch)
                    cost = criterion(prediction, target_batch)

                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()

                    writer.add_scalar('Loss/train', cost / total_batch, iter)
                    writer.add_scalar("Test/origin_accuracy", accuracy(attacked=False, val_dset=val_dset,
                                                                    model=model, test_size_limit=test_size_limit), iter)
                    writer.add_scalar("Test/advers_accuracy", accuracy(attacked=True, val_dset=val_dset,
                                                                    model=model, test_size_limit=test_size_limit), iter)
                    iter += 1

                # save weights
                torch.save(model, f"./weights/{model_arch}-{dataset}-ep{epoch:03d}-DAmageNet-{time}.pth")
                print(f"./weights/{model_arch}-{dataset}-ep{epoch:03d}-DAmageNet.pth")
# ======= DAmegeNet ==================================================================



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='VOneNet fine tunning validation Usage')
    parser.add_argument('--epochs', type=int, default=1,
                        help='how many epochs needed for fine tunning validation')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to originally trained vonenet model')
    parser.add_argument('--model_arch', type=str, required=True, default="resnet18",
                        help='arch: ["resnet18"]')
    parser.add_argument('--test_size_limit', type=int, default=None,
                        help='As test dataset is too big, we limit test_size')
    parser.add_argument('--img_size', type=int, required=True, default=224,
                        help='shape=(img_size, img_size')
    parser.add_argument('--dataset', type=str, required=True, default='imagenet',
                        help="choose in ['mnist', 'imagenet'])")
    parser.add_argument('--dset_root', type=str, required=True, default=None,
                        help="only for imagenet, parent directory of train, val dirs ex) ILSVRC2012")
    parser.add_argument('val_method', type=str, required=True, default=None,
                        help="choose in ['FGSM', 'damagenet']")
    parser.add_argument('damagenet_root', type=str, required=False, default=None,
                        help="path of parent dir consisting of DAmagenet")
    FLAGS, FIRE_FLAGS = parser.parse_known_args()

    epochs = int(FLAGS.epochs)
    batch_size = int(FLAGS.batch_size)
    learning_rate = float(FLAGS.lr)
    model_path = os.path.abspath(FLAGS.model_path)
    model_arch = FLAGS.model_arch
    test_size_limit = None if not FLAGS.test_size_limit else int(FLAGS.test_size_limit)
    image_size = (int(FLAGS.img_size), int(FLAGS.img_size))
    dataset = FLAGS.dataset
    dset_root = FLAGS.root
    val_method = FLAGS.val_method
    damagenet_root = FLAGS.damagenet_root

    logging.info(f"As resnet was trained with ImageNet dataset, image size is fixed as (224, 224)")

    logging.info(f"epochs       : {epochs}")
    logging.info(f"batch size   : {batch_size}")
    logging.info(f"learning_rate: {learning_rate}")
    logging.info(f"model_path   : {model_path}")
    logging.info(f"model_arch   : {model_arch}")
    logging.info(f"test_size    : {test_size_limit}")
    logging.info(f"image size   : {image_size}")
    logging.info(f"dataset      : {dataset}")
    logging.info(f"dset_root    : {dset_root}")

    logging.info(f"As resnet was trained with ImageNet dataset, image size is fixed as (224, 224)")

    validation(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
               model_path=model_path, model_arch=model_arch, test_size_limit=test_size_limit,
               image_size=image_size, dataset=dataset, dset_root=dset_root, val_method=val_method,
               damagenet_root=damagenet_root)

    calc_accuracy(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
               model_path=model_path, model_arch=model_arch, test_size_limit=test_size_limit,
               image_size=image_size, dataset=dataset, dset_root=dset_root)

