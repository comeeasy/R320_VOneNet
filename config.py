class ConfigTrain:
    epochs = 10
    batch_size = 64
    learning_rate = 1e-4
    model_arch = "resnet18" 
        # only resnet18 for now

    img_size = 28
        # turns into (224, 224)

    dataset = "mnist" 
        # "imagenet" or "mnist"

    dset_root = "/media/r320/2d365830-836f-4d91-8998-fef7c8443335/ImageNet_dset/ILSVRC2012"
        # dataset path not needed for mnist
        # the path of directory must consist of train and val dirs

    is_vonenet = True
        # if this var is set, train with VOneNet
        # else, train with ordinary model

    # resume options
    resume = False
    resume_model_path = "./weights/Resnet18-ImageNet-imgSize56/resnet18-imagenet-ep020-2021-11-10-16.pth"
        # if name of model weights was Resnet18-..-ep003.pth,
        # then start_epoch must be same as 3 
    start_epoch = 20

    device = 'cuda:0'
    
    # number of each of simple, complex channel
    n_simple_channel_GBF = 128
    n_complex_channel_GFB = 128
    
        
class ConfigVal:
    epochs_finetune = 50
    batch_size_finetune = 64
    learning_rate_finetune = 1e-5
    model_arch = "resnet18"

    model_path = "./weights/VOneresnet18-mnist-ImgSize28-ep010-2021-11-18-23.pth"
        # Weight of model path to validatoin
    
    is_vonenet = True
        # if this var is set, train with VOneNet
        # else, train with ordinary model
        # this is only for name of weigts, whether "VOne" should be attached or not

    img_size = 28
        # !! === NOTE === !!
        #   Must match img size to that of trained model 
        # if img_size = 224, turns into (224, 224)

    dataset = "mnist" 
        # "imagenet" or "mnist"

    dset_root = "/media/r320/2d365830-836f-4d91-8998-fef7c8443335/ImageNet_dset/ILSVRC2012"
        # dataset path not needed for mnist
        # the path of directory must consist of train and val dirs
    
    val_method = 'fgsm'
        # capable list of each dataset
        # ImageNet  : "fgsm", "damagenet"
        # MNISt     : "fgsm"
    
    damagenet_root = "/media/r320/2d365830-836f-4d91-8998-fef7c8443335/ImageNet_dset/DAmageNet"
        # the path of DAmageNet dataset,
        # which consists of DAmageNet directory 

    device = 'cuda:0'

    resume = False
    start_epoch = None
