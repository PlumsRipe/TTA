# Our on CIFAR10-C/100-C

Ours method on CIFAR-10-C/100-C under common corruptions or natural shifts. Our implementation is based on [repo](https://github.com/Yushu-Li/OWTTT) and therefore requires some similar preparation processes.

### Requirements

To install requirements:

```
pip install -r requirements.txt
```

To download datasets:

```
export DATADIR=/data/cifar
mkdir -p ${DATADIR} && cd ${DATADIR}
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
wget -O CIFAR-100-C.tar https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1
tar -xvf CIFAR-100-C.tar
wget -O tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

### Pre-trained Models

The checkpoints of pre-train Resnet-50 can be downloaded (214MB) using the following command:

```
mkdir -p results/cifar10_joint_resnet50 && cd results/cifar10_joint_resnet50
gdown https://drive.google.com/uc?id=1TWiFJY_q5uKvNr9x3Z4CiK2w9Giqk9Dx && cd ../..
mkdir -p results/cifar100_joint_resnet50 && cd results/cifar100_joint_resnet50
gdown https://drive.google.com/uc?id=1-8KNUXXVzJIPvao-GxMp2DiArYU9NBRs && cd ../..
```

These models are obtained by training on the clean CIFAR10/100 images using semi-supervised SimCLR.

### Open-Set Test-Time Adaptation:

We present our method on CIFAR10-C/100-C.

- run OURS method on CIFAR10-C under the OWTTT protocol.

  ```
  bash scripts/ours_cifar10.sh "corruption_type" "strong_ood_type" 
  ```

  Where "corruption_type" is the corruption type in CIFAR10-C, and "strong_ood_type" is the strong OOD type in [noise, MNIST, SVHN, Tiny].
  
  For example, to run OURS on CIFAR10-C under the snow corruption with MNIST as strong OOD, we can use the following command:

  ```
  bash scripts/ours_cifar10.sh snow MNIST 
  ```

  The following results are yielded by the above scripts (%) under the snow corruption, and with MNIST as strong OOD:


| Method | ACC_S | ACC_N | ACC_H |
| :------: | :-----: | :-----: | :-----: |
|  TEST  | 66.4 | 91.6 | 77.0 |
|  OWTTT  | 84.1 | 97.5 | 90.3 |
| Ours | 85.5 | 98.5 | 91.5 |
- run OURS method on CIFAR100-C under the OWTTT protocol.

  ```
  bash scripts/ours_cifar100.sh "corruption_type" "strong_ood_type" 
  ```

  Where "corruption_type" is the corruption type in CIFAR100-C, and "strong_ood_type" is the strong OOD type in [noise, MNIST, SVHN, Tiny].
  
  For example, to run OURS on CIFAR100-C under the snow corruption with MNIST as strong OOD, we can use the following command:

  ```
  bash scripts/ours_cifar100.sh snow MNIST 
  ```

  The following results are yielded by the above scripts (%) under the snow corruption, and with MNIST as strong OOD:


| Method | ACC_S | ACC_N | ACC_H |
| :------: | :-----: | :-----: | :-----: |
|  TEST  | 29.2 | 53.3 | 37.7 |
|  OWTTT  | 44.8 | 93.6 | 60.6 |
| Ours | 48.9 | 96.7 | 64.9 |

### Acknowledgements

Our code is built upon the public code of the [OWTTT](https://github.com/Yushu-Li/OWTTT).
