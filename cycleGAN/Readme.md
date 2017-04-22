# PyTorch Implementation of CycleGAN

PyTorch implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/) on the Facades dataset.

## Prerequisites
- PyTorch
- torchvision

## DATASET

  In the `CycleGAN` folder, run:
  ```
  wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz
  tar -zxvf facades.tar.gz
  rm facades.tar.gz
  ```
  Go to the `scripts` folder and run:
  ```
  python PrepareDataset.py --dataPath ../facades
  ```

  This script will split paired training image into unpaired training images. At the end of this script, it will ask you whether to delete original paired data in order to save disk space, please be aware that deleted data is unrecoverable.

## Training
  ```
  python CycleGAN.py --cuda
  ```

## Generate
  ```
  python generate.py --G_AB checkpoints/G_AB_40000.pth --G_BA checkpoints/G_BA_40000.pth -cuda --dataPath facades/val/
  ```
To train or generate on other dataset, change `dataPath` accordingly.

- Generations:

**A -> B -> A**

  <img src="samples/apple2orange/A.png" width="250"> <img src="samples/apple2orange/AB.png" width="250"> <img src="samples/apple2orange/ABA.png" width="250">

**B -> A -> B**

  <img src="samples/apple2orange/B.png" width="250"> <img src="samples/apple2orange/BA.png" width="250"> <img src="samples/apple2orange/BAB.png" width="250">

**A -> B -> A**

  <img src="samples/A.png" width="250"> <img src="samples/AB.png" width="250"> <img src="samples/ABA.png" width="250">

**B -> A -> B**

  <img src="samples/B.png" width="250"> <img src="samples/BA.png" width="250"> <img src="samples/BAB.png" width="250">

## Notes
- [DiscoGAN](https://github.com/sunshineatnoon/Paper-Implementations/tree/master/DiscoGAN) can't generate high quality reconstruction images on the Facades dataset, one thing that CycleGAN resolves this is by using a generator containing 6 residual blocks.
- It's important **not** to chain the parameters of two discriminators together, otherwise severe mode collapse when `batchSize=1` will be observed.

## Reference
1. [https://github.com/junyanz/CycleGAN](https://github.com/junyanz/CycleGAN)
2. Zhu J Y, Park T, Isola P, et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks[J]. arXiv preprint arXiv:1703.10593, 2017.
