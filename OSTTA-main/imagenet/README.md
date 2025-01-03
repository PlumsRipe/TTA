# Our on ImageNet-C

Ours method on ImageNet-C under the OWTTT protocol. Our implementation is based on [repo](https://github.com/Gorilla-Lab-SCUT/TTAC/tree/master/imagenet) and therefore requires some similar preparation processes.

### Requirements

- To install requirements:

    ```
    pip install -r requirements.txt
    ```

- To download ImageNet dataset:

    We need to firstly download the validation set and the development kit (Task 1 & 2) of ImageNet-1k on [here](https://image-net.org/challenges/LSVRC/2012/index.php), and put them under `data` folder.

- To create the corruption dataset
    ```
    python utils/create_corruption_dataset.py
    ```

    The issue `Frost missing after pip install` can be solved following [here](https://github.com/hendrycks/robustness/issues/4#issuecomment-427226016).

    Finally, the structure of the `data` folder should be like
    ```
    data
    |_ ILSVRC2012_devkit_t12.tar
    |_ ILSVRC2012_img_val.tar
    |_ val
        |_ n01440764
        |_ ...
    |_ imagenet-r
        |_ n01443537
        |_ ...
    |_ corruption
        |_ brightness.pth
        |_ contrast.pth
        |_ ...
    |_ meta.bin
    ```

### Pre-trained Models

Here, we use the pretrain model provided by torchvision.

### Open-Set Test-Time Adaptation:

We present our method on ImageNet-C.

- run OURS method or the baseline method TEST on ImageNet-C under the OWTTT protocol.

    ```
    bash scripts/ours_c.sh "corruption_type" "strong_ood_type" 
    ```
    Where "corruption_type" is the corruption type in ImageNet-C, and "strong_ood_type" is the strong OOD type in [noise, MNIST, SVHN]. 
    
    For example, to run OURS on ImageNet-C under the snow corruption with MNIST as strong OOD, we can use the following command:
    
    ```
    bash scripts/ours_c.sh snow MNIST 
    ```

    The following results are yielded by the above scripts (%) under the snow corruption, and with MNIST as strong OOD:
    
    | Method | ACC_S | ACC_N | ACC_H |
    |:------:|:-------:|:-------:|:-------:|
    |  TEST  |   17.3   |   99.4   |  29.5  |
    |  OWTTT  |   45.3    |    100.0   | 62.4 |
    | Ours | 47.2 | 99.8 | 64.1 |


### Acknowledgements

Our code is built upon the public code of the [OWTTT](https://github.com/Yushu-Li/OWTTT).
