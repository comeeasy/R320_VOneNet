from scipy.stats.stats import weightedtau
from utils.adversarial_attack import generate_image_adversary
from tqdm import tqdm
from config import ConfigVal

import torch
import torch.utils.tensorboard as tensorboard
import utils.data as data
import time


def accuracy(img_batch, target_batch, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        X_test = img_batch.to(device)
        Y_test = target_batch.to(device)

        prediction = model(X_test).to(device)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()

    return accuracy

class Fgsm:
    """
    class for FGSM validation method
    """
    def calc_accuracy(model, val_dset, writer, epoch):
        """
        calculating accuracy of total test dataset
        and save result on tensorboard

        == execute below on terminal ==
        $ tensorboard --logdir runs/ --bind_all

        """

        iter = 0
        clear_accuracy_avg = 0
        adv_accuracy_avg = 0
        total_batch = len(val_dset)
        device = ConfigVal.device

        for img_batch, target_batch in tqdm(val_dset):
            img_batch = img_batch.to(device)
            writer.add_images("Images/original image batch", img_batch, iter)

            adv_img_batch, adv_target_batch = generate_image_adversary(
                model=model, 
                img_batch=img_batch,
                target_batch=target_batch)
            writer.add_images("Images/adversarial image batch", adv_img_batch, iter)

            adv_accuracy_avg += accuracy(adv_img_batch, adv_target_batch, model) / total_batch
            clear_accuracy_avg += accuracy(img_batch, target_batch, model) / total_batch

            iter += 1
        
        writer.add_scalar("Test/advers_accuracy", adv_accuracy_avg, epoch)
        writer.add_scalar("Test/origin_accuracy", clear_accuracy_avg, epoch)

    def finetune(model, epochs, train_dset, dataset, criterion, optimizer, model_arch, val_dset, writer):
        """
        finetuning with FGSM method.
        result: 
            ==== execute below on terminal ====
            $ tensorboard --logdir runs/ --bind_all

            images:
                1. clear image                          (Image/original image batch)
                2. adversarial image                    (Image/adversarial image batch)
            batch accuracy
                1. clear accuracy                       (Train/origin_batch_accuracy)
                2. adversarial accuracy                 (Train/advers_batch_accuracy)
            total accuracy of testset (calc by val dset)
                1. clear accuracy                       (Test/origin_accuracy)
                2. adversarial accuracy                 (Test/advers_accuracy)
            Loss
                1. loss value during training           (Train/loss)
            weights
                for every epoch, weight file(*.pth) will be saved
                please check weights directory
                ex ) VOneresnet18-imagenet-ep002-2021-11-07-14.pth

        """

        iter = 0

        for epoch in range(1, epochs + 1):
            for img_batch, target_batch in tqdm(train_dset):
                writer.add_images("Images/original image batch", img_batch, iter)

                if dataset.lower() == 'imagenet':
                    adv_img_batch, adv_target_batch = generate_image_adversary(
                        model=model, 
                        img_batch=img_batch,
                        target_batch=target_batch)
                writer.add_images("Images/adversarial image batch", adv_img_batch, iter)

                prediction = model(adv_img_batch)
                cost = criterion(prediction, adv_target_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                clear_batch_acc = accuracy(img_batch=img_batch, target_batch=target_batch, model=model)
                adv_batch_acc = accuracy(img_batch=img_batch, target_batch=target_batch, model=model)

                writer.add_scalar('Train/loss', cost, iter)
                writer.add_scalar("Train/origin_batch_accuracy", clear_batch_acc)
                writer.add_scalar("Train/advers_batch_accuracy", adv_batch_acc)

                iter += 1
                
            Fgsm.calc_accuracy(model, val_dset, epoch)

            # save weights
            weight_path = f"./weights/{model_arch}-{dataset}-ep{epoch:03d}-FGSM-{time.strftime('%Y-%m-%d-%H')}.pth"
            torch.save(model, weight_path)
            print(weight_path)

class DamageNet:
    """
    class for DAmageNet (black-box attack) validation method
    """

    def calc_accuracy(model, damagenet_root, imagenet_root, image_size, batch_size, writer, epoch):
        """
        calculating accuracy of total test dataset
        and save result on tensorboard

        == execute below on terminal ==
        $ tensorboard --logdir runs/ --bind_all

        """


        damagenet_val = data.get_damegenet(root=damagenet_root, img_size=image_size, batch_size=batch_size)
        _, imagenet_val = data.get_imagenet(root=imagenet_root, img_size=image_size, batch_size=batch_size)

        iter = 0
        adv_accuracy_avg = 0
        for img_batch, target_batch in tqdm(damagenet_val):
            writer.add_images("Images/adversarial image batch", img_batch, iter)
            adv_accuracy_avg += accuracy(img_batch, target_batch, model)
            iter += 1
        
        iter = 0
        clear_accuracy_avg = 0
        for img_batch, target_batch in tqdm(imagenet_val):
            writer.add_images("Images/original image batch", img_batch, iter)
            clear_accuracy_avg += accuracy(img_batch, target_batch, model)

        writer.add_scalar('Test/advers_accuracy', adv_accuracy_avg, epoch)
        writer.add_scalar("Test/origin_accuracy", clear_accuracy_avg, epoch)



    def finetune(model, model_arch, epochs, damagenet_root, dset_root, dataset, image_size, batch_size, criterion, optimizer, writer):
        """
        finetuning with DAmageNet (black-box attack) method.
        result: 
            ==== execute below on terminal ====
            $ tensorboard --logdir runs/ --bind_all

            images:
                1. clear image                          (Image/original image batch)
                2. adversarial image                    (Image/adversarial image batch)
            batch accuracy
                1. clear accuracy                       (Train/origin_batch_accuracy)
                2. adversarial accuracy                 (Train/advers_batch_accuracy)
            total accuracy of testset (calc by val dset)
                1. clear accuracy                       (Test/origin_accuracy)
                2. adversarial accuracy                 (Test/advers_accuracy)
            Loss
                1. loss value during training           (Train/loss)
            weights
                for every epoch, weight file(*.pth) will be saved
                please check weights directory
                ex ) VOneresnet18-imagenet-ep002-2021-11-07-14.pth

        """

        damagenet_val = data.get_damegenet(root=damagenet_root, img_size=image_size, batch_size=batch_size)
        imagenet_folder = data.get_imagenet_folder(root=dset_root, img_size=image_size)

        device = ConfigVal.device

        iter = 0

        for epoch in range(1, epochs+1):
            total_batch = len(damagenet_val)

            clear_idx = 0
            for adv_img_batch, adv_target_batch in tqdm(damagenet_val):
                # get clear images as batch
                clear_img_batch = [imagenet_folder.__getitem__(i + clear_idx)[0].unsqueeze(0) for i in range(batch_size)]
                clear_img_batch = torch.cat(clear_img_batch, dim=0)
                clear_target_batch = [imagenet_folder.__getitem__(i + clear_idx)[1] for i in range(batch_size)]
                clear_target_batch = torch.Tensor(clear_target_batch)
                clear_idx += batch_size
                
                writer.add_images("Images/clear image batch", clear_img_batch, iter)
                writer.add_images("Images/adversarial image batch", adv_img_batch, iter)
                
                clear_img_batch = clear_img_batch.to(device)
                clear_target_batch =clear_target_batch.to(device)
                adv_img_batch = adv_img_batch.to(device)
                adv_target_batch = adv_target_batch.to(device)

                prediction = model(adv_img_batch)
                cost = criterion(prediction, adv_target_batch)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                writer.add_scalar('Loss/train', cost / total_batch, iter)
                writer.add_scalar("Train/origin_batch_accuracy", accuracy(clear_img_batch, clear_target_batch,model), iter)
                writer.add_scalar("Train/advers_batch_accuracy", accuracy(adv_img_batch, adv_target_batch, model), iter)
                iter += 1

            DamageNet.calc_accuracy(model=model, damagenet_root=damagenet_root, imagenet_root=dset_root,
                                    image_size=image_size, batch_size=batch_size, epoch=epoch)

            # save weights
            weight_path = f"./weights/{model_arch}-{dataset}-ep{epoch:03d}-DAmageNet-{time.strftime('%Y-%m-%d-%H')}.pth"
            torch.save(model, weight_path)
            print(weight_path)