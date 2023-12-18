# StyleGAN2

This is an implementation of StyleGAN2 for study purpose.

This is also a part of my homework for MachineLearning Course by Hung-yi Lee (NTU).

The original paper can be found here: https://github.com/NVlabs/stylegan2

The implementation I refer to is: https://github.com/lucidrains/stylegan2-pytorch

## Installation

    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install ipykernel
    pip install accelerate
    pip install opencv-python
    pip install matplotlib

## Training

Please prepare a set of images in the /dataset folder or a pickle file containing a list of training RGB images.

Next, run the Train_StyleGAN2.ipynb for training.