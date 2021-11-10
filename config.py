class ConfigTrain:
    epochs = 10
    batch_size = 8
    learning_rate = 1e-3
    model_arch = "resnet18" 
        # only resnet18 for now

    img_size = 224 
        # turns into (224, 224)

    dataset = "imagenet" 
        # "imagenet" or "mnist"

    dset_root = "/media/r320/2d365830-836f-4d91-8998-fef7c8443335/ImageNet_dset/ILSVRC2012"
        # dataset path not needed for mnist
        # the path of directory must consist of train and val dirs

    is_vonenet = False 
        # if this var is set, train with VOneNet
        # else, train with ordinary model

    device = 'cuda:0'
    
        
class ConfigVal:
    epochs_finetune = 1
    batch_size_finetune = 4
    learning_rate_finetune = 1e-4
    model_arch = "resnet18"

    model_path = "./weights/VOneResnet18-imagenet-ImgSize224/VOneresnet18-imagenet-ep009-2021-11-09-16.pth"
        # Weight of model path to validatoin
    
    img_size = 224
        # !! === NOTE === !!
        #   Must match img size to that of trained model 
        # turns into (224, 224)

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