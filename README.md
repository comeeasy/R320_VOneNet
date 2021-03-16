# FGSM_MNIST

We're trying to make model better which is robust against to adversarial images, especially made by FGSM.
Yann LeCun's MNIST datasets are used.

We're inspired by this [tutorial](https://www.pyimagesearch.com/2021/03/08/defending-against-adversarial-image-attacks-with-keras-and-tensorflow/).

A function named
  generate_image_adversarial(args) is just interpretation of tensorflow code of above tutorial

## Requirements

- python 3.8+
- pytorch 0.4.1+
- numpy
- tqdm

## License

GNU GPL 3+

## trained model

| Name     | Description                                                              |
| -------- | ------------------------------------------------------------------------ |
| 1-layer-linear-classifier | really simple model                                     |
| 3-layer-linear-classifier | add two layer to 1-layer simple model                   |
| Convnet                   | simple convolutional model                              |

## Longer Motivation

1. VOneNet maybe boosts performance. So we're considering how apply this model
[VOneNet](https://github.com/dicarlolab/vonenet)
