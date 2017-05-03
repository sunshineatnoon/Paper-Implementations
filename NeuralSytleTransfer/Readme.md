# PyTorch Implementation of [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)

**This code heavily borrows from [https://github.com/leongatys/PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer), which is an elegant example to show how to extract intermediate features from pre-trained models in PyTorch**

## Prerequisites
- PyTorch
- torchvision

## Downloading Pre-trained VGG model
  ```
  mkdir models && cd models
  wget -c --no-check-certificate https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth
  cd ..
  ```
## Training
  ```
  python neuralStyle.py --cuda
  ```
**Train on CPU: leave out the `--cuda` parameter**

Transferred image will be stored as `images/transfer.png`

<img src="images/dancing.jpg" height="200"> <img src="images/picasso.jpg" height="200"> <img src="images/transfer_dancing.png" height="200">

<img src="images/corgi.jpg" height="200"> <img src="images/candy.jpg" height="200"> <img src="images/transfer_corgi.png" height="200">

## Reference
1. [https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)
2. [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
