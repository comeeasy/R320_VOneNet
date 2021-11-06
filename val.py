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


def accuracy(img_batch, target_batch, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        X_test = img_batch.to(device)
        Y_test = target_batch.to(device)

        prediction = model(X_test).to(device)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()

    return accuracy

def calc_accuracy(batch_size, model_path, image_size, dataset, 
                  dset_root, val_method, damagenet_root):
    
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"{model_path} is not exist")

    logging.info(f"Getting initial Accuracy of {model_path} trained by {dataset}")
    logging.info(f"accuracy of original data, adversarial data will be written at tensorboard. please check latest data if runs directory.")
    logging.info(f"Usage: $ tensorboard --logdir ./runs --bind_all")

    logging.info(f"loading dataset {dataset}")
    if dataset.lower() == 'mnist':
        train_dset, val_dset = data.get_mnist(batch_size=batch_size, image_size=image_size)
    elif dataset.lower() == 'imagenet':
        train_dset, val_dset = data.get_imagenet(root=dset_root, img_size=image_size, batch_size=batch_size)

    if val_method.lower() == 'damagenet':
        if dataset.lower() != 'imagenet':
            raise RuntimeError("damagenet validation method only supports imagenet")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Use {device}")

    logging.info(f"loading model {model_path}")
    model = torch.load(model_path)
    model.eval().to(device)
    
    with tensorboard.SummaryWriter() as writer:
        iter = 0
# ======= FGSM =======================================================================
        if val_method.lower() == 'fgsm':
            total_batch = len(val_dset)

            clear_accuracy_avg = 0
            adv_accuracy_avg = 0
            for img_batch, target_batch in tqdm(val_dset):
                adv_img_batch, adv_target_batch = generate_image_adversary(
                    model=model, 
                    img_batch=img_batch,
                    target_batch=target_batch)
                writer.add_images("Images/adversarial image batch", adv_img_batch, iter)

                adv_accuracy_avg += accuracy(adv_img_batch, adv_target_batch, model) / total_batch
                clear_accuracy_avg += accuracy(img_batch, target_batch, model) / total_batch
            
            writer.add_scalar("Test/advers_accuracy", adv_accuracy_avg)
            writer.add_scalar("Test/origin_accuracy", clear_accuracy_avg)
# ======= FGSM =======================================================================
# ======= DAmegeNet ==================================================================
        elif val_method.lower() == 'damagenet':
            if dataset.lower() != 'imagenet':
                raise RuntimeError("To validate with DAmageNet, must choose imagenet as dataset")

            damagenet_val = data.get_damegenet(root=damagenet_root, img_size=image_size, batch_size=batch_size)
            _, imagenet_val = data.get_imagenet(root=dset_root, img_size=image_size, batch_size=batch_size)

            total_batch = len(damagenet_val)

            adv_accuracy_avg = 0
            for img_batch, target_batch in tqdm(damagenet_val):
                writer.add_images("Images/adversarial image batch", img_batch, iter)
                adv_accuracy_avg += accuracy(img_batch, target_batch, model)
            
            clear_accuracy_avg = 0
            for img_batch, target_batch in tqdm(imagenet_val):
                writer.add_images("Images/original image batch", img_batch, iter)
                clear_accuracy_avg += accuracy(img_batch, target_batch, model)

            writer.add_scalar('Test/advers_accuracy', adv_accuracy_avg)
            writer.add_scalar("Test/origin_accuracy", clear_accuracy_avg)
# ======= DAmegeNet ==================================================================


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
            imagenet_folder = data.get_imagenet_folder(root=dset_root, img_size=image_size)

            for epoch in range(epochs):
                total_batch = len(damagenet_val)

                clear_idx = 0
                for img_batch, target_batch in tqdm(damagenet_val):
                    # get clear images as batch
                    clear_img_batch = [imagenet_folder.__getitem__(i + clear_idx)[0].unsqueeze(0) for i in range(batch_size)]
                    clear_img_batch = torch.cat(clear_img_batch, dim=0)
                    clear_target_batch = [imagenet_folder.__getitem__(i + clear_idx)[1] for i in range(batch_size)]
                    clear_target_batch = torch.Tensor(clear_target_batch)
                    clear_idx += batch_size
                    
                    writer.add_images("Images/clear image batch", clear_img_batch, iter)
                    writer.add_images("Images/adversarial image batch", img_batch, iter)
                   
                    clear_img_batch = clear_img_batch.to(device)
                    img_batch = img_batch.to(device)
                    target_batch = target_batch.to(device)

                    prediction = model(img_batch)
                    cost = criterion(prediction, target_batch)

                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()

                    writer.add_scalar('Loss/train', cost / total_batch, iter)
                    writer.add_scalar("Test/origin_accuracy", accuracy(
                        clear_img_batch, 
                        clear_target_batch,
                        model), iter)
                    writer.add_scalar("Test/advers_accuracy", accuracy(
                        img_batch, 
                        target_batch,                                           
                        model), iter)
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
    parser.add_argument('--val_method', type=str, required=True, default=None,
                        help="choose in ['FGSM', 'damagenet']")
    parser.add_argument('--damagenet_root', type=str, required=False, default=None,
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
    dset_root = FLAGS.dset_root
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
    
    if dataset.lower() == 'damagenet':
        if not os.path.exists(damagenet_root):
            raise FileNotFoundError("DAmageNet path")
        logging.info(f"damagenet_root   : {damagenet_root}")


    logging.info(f"As resnet was trained with ImageNet dataset, image size is fixed as (224, 224)")

    #calc_accuracy(batch_size=batch_size, model_path=model_path, image_size=image_size, dataset=dataset, 
    #              dset_root=dset_root, val_method=val_method, damagenet_root=damagenet_root)
    
    validation(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
               model_path=model_path, model_arch=model_arch, test_size_limit=test_size_limit,
               image_size=image_size, dataset=dataset, dset_root=dset_root, val_method=val_method,
               damagenet_root=damagenet_root)
