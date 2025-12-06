#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   prepper.py
# @Desc     :   

from random import randint

from pipeline.processor import process_data

from src.configs.cfg_base4dl import CONFIG4DL
from src.dataloaders.dataloader4torch import TorchDataLoader


def prepare_data() -> tuple[TorchDataLoader, TorchDataLoader, int]:
    """ Prepare data """
    # Get dataset
    train, valid, MAX_SEQ_LEN = process_data()

    # Set up dataloader
    dataloader4train = TorchDataLoader(
        train,
        batch_size=CONFIG4DL.PREPROCESSOR.BATCHES,
        shuffle_state=CONFIG4DL.PREPROCESSOR.SHUFFLE,
        workers=CONFIG4DL.PREPROCESSOR.WORKERS,
    )
    dataloader4valid = TorchDataLoader(
        valid,
        batch_size=CONFIG4DL.PREPROCESSOR.BATCHES,
        shuffle_state=CONFIG4DL.PREPROCESSOR.SHUFFLE,
        workers=CONFIG4DL.PREPROCESSOR.WORKERS,
    )
    # idx4train: int = randint(0, len(dataloader4train) - 1)
    # print(dataloader4train[idx4train])
    # print(dataloader4valid[idx4train])

    return dataloader4train, dataloader4valid, MAX_SEQ_LEN


if __name__ == "__main__":
    prepare_data()
