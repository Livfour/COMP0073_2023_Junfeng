# Environment Setting
The python version is 3.10.12.
Your can try to use `reuqirements.txt` to install the python packages, but it is may does not work due to some packages cannot be install by pip or conda.

## Set the python environment manually
1. Install pytorch: the pytorch version used in this prject is 2.0.1. Please follow the instruction on https://pytorch.org/get-started/locally/.
2. Install mmcv: the mmcv version used in this project is 2.0.1. Please follow the instruction on https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-mmcv.
3. Install other package by conda or pip:
    + numpy
    + tqdm
    + timm
    + albumentations
    + cv2
    + matplotlib
4. Download the dataset: follow the instruction on https://github.com/anDoer/PoseTrack21.

## Download pretrined weight
Allw weight should be put in `checkpoints` folder.
1. ViTPose backbone
The ViTPose backbone weight must be downloaded. You can download this weight here: https://1drv.ms/u/s!AimBgYV7JjTlgcccwaTZ8xCFFM3Sjg?e=chmiK5. More weight could be found on their github page: https://github.com/ViTAE-Transformer/ViTPose. The paper could be found: https://arxiv.org/pdf/2204.12484.pdf.
2. For weight trained in this project could be found on https://drive.google.com/drive/folders/1vPEjDVbE4wztwpybQItXpaX7M4Xz9YLy?usp=sharing.

# Training
Just use python3 to run the train_**.py code, but you need to modify the code to chage the `dataset_root_dir` to your dataset root directory.

# Testing
You can run the jupyter file in `test` folder to run the test, or convert it into python file to run it, make sure that the work dir is the root dir. The result could be found in `results` folder.