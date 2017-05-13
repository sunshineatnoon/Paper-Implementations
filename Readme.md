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
## Neural Style Transfer 
  ```
  python neuralStyle.py --cuda
  ```
  <img src="images/dancing.jpg" height="200"> <img src="images/picasso.jpg" height="200"> <img src="images/transfer_dancing.png" height="200">

  <img src="images/corgi.jpg" height="200"> <img src="images/candy.jpg" height="200"> <img src="images/transfer_corgi.png" height="200">
## Neural Style Transfer with Color Preservation

   This implements [Preserving Color in Neural Artistic Style Transfer](https://arxiv.org/abs/1606.05897). Color Histogram Transfer algorithm is copied from [chainer-neural-style](https://github.com/dsanno/chainer-neural-style).
   
   ```
   python train.py --style_image images/picasso.jpg --content_image images/NY.png --content_weight 500 --style_weight 1 --cuda --color_histogram_matching
   ```
   <img src="images/NY.png" height="200"> <img src="images/picasso.jpg" height="200"> <img src="images/NY_transfer.png" height="200">

**Train on CPU: leave out the `--cuda` parameter**

Transferred image will be stored as `images/transfer.png`

<<<<<<< HEAD

## Reference
1. [https://github.com/yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)
2. [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
3. [https://github.com/dsanno/chainer-neural-style](https://github.com/dsanno/chainer-neural-style)
=======
<img src="images/dancing.jpg" height="200"> <img src="images/picasso.jpg" height="200"> <img src="images/dancing_transfer.png" height="200">

<img src="images/corgi.jpg" height="200"> <img src="images/candy.jpg" height="200"> <img src="images/transfer_corgi.png" height="200">

## Reference
1. [https://github.com/leongatys/PytorchNeuralStyleTransfer](https://github.com/leongatys/PytorchNeuralStyleTransfer)
2. [https://github.com/alexis-jacq/Pytorch-Tutorials](https://github.com/alexis-jacq/Pytorch-Tutorials)
>>>>>>> fd4214a03f98585d25f68da81fd7bae671a4e8cf
