#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/2 22:23
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   trainer.py
# @Desc     :   

from pathlib import Path
from torch import optim, nn

from pipeline.prepper import prepare_data

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.parser import set_argument_parser
from src.trainers.trainer4torch import TorchTrainer
from src.nets.rnn4classification import NormalRNNForClassification
from src.utils.stats import load_json
from src.utils.PT import TorchRandomSeed


def main() -> None:
    """ Main Function """
    # Set up argument parser
    args = set_argument_parser()

    with TorchRandomSeed("Sequence to one - Next Word Prediction"):
        # Get the dictionary
        dic: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY)
        dictionary = load_json(dic)

        # Get the data
        train_loader, valid_loader, MAX_SEQ_LEN = prepare_data()

        # Get the input size and number of classes
        vocab_size: int = len(dictionary)

        # Initialize model
        model = NormalRNNForClassification(
            vocab_size=vocab_size,
            embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
            hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
            num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
            num_classes=vocab_size,
            dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
            accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
        )
        # Setup optimizer and loss function
        optimizer = optim.AdamW(model.parameters(), lr=args.alpha, weight_decay=CONFIG4RNN.HYPERPARAMETERS.DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss()
        model.summary()

        # Setup trainer
        trainer = TorchTrainer(
            model=model,
            optimiser=optimizer,
            criterion=criterion,
            accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
            scheduler=scheduler,
        )
        # Train the model
        trainer.fit(
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=args.epochs,
            model_save_path=str(CONFIG4RNN.FILEPATHS.SAVED_NET),
            log_name="THULAC"
        )


if __name__ == "__main__":
    main()
