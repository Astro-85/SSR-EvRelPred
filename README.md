# In Defense of Structural Symbolic Representation for Event-Relation Prediction (L3D-IVU Workshop, CVPR23)

[![LICENSE](https://img.shields.io/badge/license-MIT-green)]
[![Python](https://img.shields.io/badge/python-3.6-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.5-yellow)
[![Arxiv](https://img.shields.io/badge/Arxiv-2301.03410-purple)](https://arxiv.org/abs/2301.03410)

**[In Defense of Structural Symbolic Representation for Event-Relation Prediction](https://arxiv.org/abs/2301.03410)**<br>
Andrew Lu*, Xudong Lin*, Yulei Niu, Shih-Fu Chang

This repository includes:

1. Instructions to install, download and process VidSitu Dataset.
2. Code to run all experiments provided in the paper along with log files.

# Download

Please see https://github.com/TheShadow29/VidSitu/blob/main/data/DATA_PREP.md for detailed instructions on downloading and setting up the dataset.

# Installation

Please see [INSTALL.md](./INSTALL.md) for detailed instructions


# Training

- Basic usage is `CUDA_VISIBLE_DEVICES=$GPUS python train_vidsitu.py "experiment_name" --arg1=val1 --arg2=val2` and the arg1, arg2 can be found in `configs/vsitu_cfg.yml`.

- For training RoBERTa models using only text annotations, no video features need to be downloaded. We include text annotations for VidSitu and VisualComet in `data/`. Please refer to the [original VidSitu repository](https://github.com/TheShadow29/VidSitu/tree/main) for more details regarding downloading video features and running experiments for verb and argument prediction. 

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

- Eg. Training command for contextualized event-sequence RoBERTa using VidSitu dataset `CUDA_VISIBLE_DEVICES=0 python train_vidsitu.py "experiment_name" --task_type='evrel' --mdl.mdl_name='rob_evrel' --train.bs=8 --train.bsv=8 --train.nw=4 --train.nwv=4 --ds.vsitu.evrel_trimmed=False --train.event_sequence=True`

- Eg. Training command for contextualized event-sequence RoBERTa using VisualComet dataset. Currently only single gpu training supported. `CUDA_VISIBLE_DEVICES=0 python train_viscom.py "experiment_name" --task_type='evrel' --mdl.mdl_name='rob_evrel' --train.bs=8 --train.bsv=8 --train.nw=4 --train.nwv=4`




# Logging

Logs are stored inside `tmp/` directory. When you run the code with $exp_name the following are stored:
- `txt_logs/$exp_name.txt`: the config used and the training, validation losses after ever epoch.
- `models/$exp_name.pth`: the model, optimizer, scheduler, accuracy, number of epochs and iterations completed are stored. Only the best model upto the current epoch is stored.
- `ext_logs/$exp_name.txt`: this uses the `logging` module of python to store the `logger.debug` outputs printed. Mainly used for debugging.
- To quickly clear all files from a previous run, use `bash clean.sh $exp_name`
- `predictions`: the validation outputs of current best model.
