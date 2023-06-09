# In Defense of Structural Symbolic Representation for Event-Relation Prediction (L3D-IVU Workshop, CVPR23)

[![LICENSE](https://img.shields.io/badge/license-MIT-green)]
[![Python](https://img.shields.io/badge/python-3.6-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.5-yellow)
[![Arxiv](https://img.shields.io/badge/Arxiv-2301.03410-purple)](https://arxiv.org/abs/2301.03410)

**[In Defense of Structural Symbolic Representation for Event-Relation Prediction](https://arxiv.org/abs/2301.03410)**<br>
Andrew Lu*, Xudong Lin*, Yulei Niu, Shih-Fu Chang

This repository includes:

NOTE: Some necessary modules are temporarily unavailible but will be added in the coming days!

1. Instructions to install, download and process VidSitu Dataset.
2. Code to run all experiments provided in the paper along with log files.

# Download

Please see [DATA_PREP.md](./data/DATA_PREP.md) for detailed instructions on downloading and setting up the dataset.

# Installation

Please see [INSTALL.md](./INSTALL.md) for detailed instructions


# Training

- Basic usage is `CUDA_VISIBLE_DEVICES=$GPUS python main_dist.py "experiment_name" --arg1=val1 --arg2=val2` and the arg1, arg2 can be found in `configs/vsitu_cfg.yml`.

- Set `$GPUS=0` for single gpu training. For multi-gpu training via Pytorch Distributed Data Parallel use `$GPUS=0,1,2,3`

- YML has a hierarchical structure which is supported using `.`
    For instance, if you want to change the `beam_size` under `gen` which in the YML file looks like
    ```
    gen:
        beam_size: 1
    ```
    you can pass `--gen.beam_size=5`

- Sometimes it might be easier to directly change the default setting in `configs/vsitu_cfg.yml` itself.

- To keep the code modular, some configurations are set in `code/extended_config.py` as well.

- All model choices are available under `code/mdl_selector.py`

# Logging

Logs are stored inside `tmp/` directory. When you run the code with $exp_name the following are stored:
- `txt_logs/$exp_name.txt`: the config used and the training, validation losses after ever epoch.
- `models/$exp_name.pth`: the model, optimizer, scheduler, accuracy, number of epochs and iterations completed are stored. Only the best model upto the current epoch is stored.
- `ext_logs/$exp_name.txt`: this uses the `logging` module of python to store the `logger.debug` outputs printed. Mainly used for debugging.
- `predictions`: the validation outputs of current best model.
