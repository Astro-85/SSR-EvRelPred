# Installation

We provide instructions to install the required dependencies.

Requirements:
+ python>=3.6
+ pytorch==1.5 (should work with pytorch >=1.5 as well but not tested)

1. Clone the repo:
    ```
    git clone https://github.com/Astro-85/SSR-EvRelPred.git
    cd VidSitu
    export ROOT=$(pwd)
    ```

1. To use the same environment you can use conda and the environment file vsitu_pyt_env.yml file provided.
    ```
    conda env create -f vsitu_pyt_env.yml
    conda activate vsitu_pyt
    ```

1. Install submodules:

    + Install Detectron2 (needed for SlowFast). If you have CUDA 10.2 and Pytorch 1.5 you can use:
    ```
    python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html
    ```
    Please refer to [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) repository for more details.

    + Slowfast:
    ```
    cd $ROOT/SlowFast
    git checkout d4a4acde2bd344316789bbd6fa778e37852f522f
    python setup.py build develop
    ```

    + Fairseq:
    ```
    cd $ROOT/fairseq
    git checkout 6fe74e50741b08a85dc203ac26d551323fd5c82d
    pip install -e .
    ```

    + cocoapi:
    ```
    cd $ROOT/cocoapi/PythonAPI
    make
    ```

    + coco-caption: (NOTE: You may need to install java). No additional steps are needed.

    Alternatively, you can run `bash $ROOT/scripts/install_all.sh`

    + coval:
    ```
    cd $ROOT/coval
    pip install -e .
    ```