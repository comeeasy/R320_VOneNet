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
        
class ConfigVal:
    epochs_finetune = 1
    batch_size_finetune = 8
    learning_rate_finetune = 1e-4
    model_arch = "resnet18"

    model_path = "./weights/Resnet18-ImageNet-imgSize56/resnet18-imagenet-ep020-2021-11-10-16.pth"
        # Weight of model path to validatoin

    test_size = 5   
        # Calculating the whole test dataset consumes much time.
        # So selected some batches from test datset as many as test_size.
        # if you want to use it, set "shuffle" arguments,
        # which is of the test dataset as True.
    
    img_size = 56
        # !! === NOTE === !!
        #   Must match img size to that of trained model 
        # turns into (224, 224)

    dataset = "imagenet" 
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