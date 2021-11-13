class ConfigTrain:
    epochs = 30
    batch_size = 64
    learning_rate = 1e-4
    model_arch = "resnet18" 
        # only resnet18 for now

    img_size = 56 
        # turns into (224, 224)

    dataset = "imagenet" 
        # "imagenet" or "mnist"

    dset_root = "/media/r320/2d365830-836f-4d91-8998-fef7c8443335/ImageNet_dset/ILSVRC2012"
        # dataset path not needed for mnist
        # the path of directory must consist of train and val dirs

    is_vonenet = False 
        # if this var is set, train with VOneNet
        # else, train with ordinary model

    # resume options
    resume = True
    resume_model_path = "./weights/Resnet18-ImageNet-imgSize56/resnet18-imagenet-ep020-2021-11-10-16.pth"
        # if name of model weights was Resnet18-..-ep003.pth,
        # then start_epoch must be same as 3 
    start_epoch = 20

    device = 'cuda:0'
    
        
class ConfigVal:
    epochs_finetune = 10
    batch_size_finetune = 16
    learning_rate_finetune = 1e-5
    model_arch = "resnet18"

    model_path = "./weights/Resnet18-ImageNet-imgSize56/resnet18-imagenet-ep020-2021-11-10-16.pth"
        # Weight of model path to validatoin
    
    img_size = 56
        # !! === NOTE === !!
        #   Must match img size to that of trained model 
        # if img_size = 224, turns into (224, 224)

    dataset = "imagenet" 
        # "imagenet" or "mnist"

    dset_root = "/media/r320/2d365830-836f-4d91-8998-fef7c8443335/ImageNet_dset/ILSVRC2012"
        # dataset path not needed for mnist
        # the path of directory must consist of train and val dirs
    
    val_method = 'damagenet'
        # capable list of each dataset
        # ImageNet  : "fgsm", "damagenet"
        # MNISt     : "fgsm"
    
    damagenet_root = "/media/r320/2d365830-836f-4d91-8998-fef7c8443335/ImageNet_dset/DAmageNet"
        # the path of DAmageNet dataset,
        # which consists of DAmageNet directory 

    device = 'cuda:0'

    resume = False
    start_epoch = None
