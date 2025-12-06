#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   evaluator.py
# @Desc     :

from pathlib import Path
from re import match
from random import randint
from torch import Tensor, load, device, no_grad, nn
from tqdm import tqdm

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import SeqTaskMode
from src.dataloaders.dataloader4torch import TorchDataLoader
from src.datasets.seq_next_step import TorchDataset4SeqPredictionNextStep
from src.nets.rnn4classification import NormalRNNForClassification
from src.utils.helper import Timer
from src.utils.highlighter import starts, lines, red, green
from src.utils.SQL import SQLiteIII
from src.utils.stats import create_full_data_split, load_json
from src.utils.THU import cut_only


def main() -> None:
    """ Main Function """
    # Get the data from the database: Method II
    with SQLiteIII(CONFIG4RNN.DATABASE.TABLE, CONFIG4RNN.DATABASE.COL) as db:
        data = db.get_all_data()
        # print(len(data))
        # print()

    with Timer("Next Word Prediction"):
        # Separate the data
        _, _, sentences = create_full_data_split(data)

        # Set a dictionary
        # amount: int | None = 100
        amount: int | None = None
        items: list[str] = []
        if amount is None:
            for line in tqdm(sentences, total=len(sentences), desc="Tokenizing Train Data"):
                for item in cut_only(line):
                    if item in CONFIG4RNN.PUNCTUATIONS.CN or match(r"^[\u4e00-\u9fff]+$", item):
                        items.append(item)
        else:
            for line in tqdm(sentences[:amount], total=amount, desc="Tokenizing Train Data"):
                for item in cut_only(line):
                    if item in CONFIG4RNN.PUNCTUATIONS.CN or match(r"^[\u4e00-\u9fff]+$", item):
                        items.append(item)
        # print(items)
        # print()

        # Load the dictionary and convert
        dic: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY)
        dictionary: dict[str, int] = load_json(dic)
        reversed_dictionary: dict = {idx: word for word, idx in dictionary.items()}

        # Convert the whole sentence to sequence using dictionary
        UNK_IDX: int = dictionary["<UNK>"]
        sequence: list[int] = [dictionary.get(item, UNK_IDX) for item in items]
        # print(sequence)
        # print()

        # Set up dataset
        dataset = TorchDataset4SeqPredictionNextStep(
            sequence,
            seq_max_len=CONFIG4RNN.PREPROCESSOR.MAX_SEQUENCE_LEN,
            mode=SeqTaskMode.SEQ2ONE,
            pad_token=dictionary["<PAD>"]
        )
        starts()
        print("Data Preprocessing Summary:")
        lines()
        print(f"Test dataset: {len(dataset)} Samples")
        print(f"Dictionary Size: {len(dictionary)}")
        print(f"Reversed Dictionary Size: {len(reversed_dictionary)}")
        starts()
        print()
        # Set up dataloader
        dataloader = TorchDataLoader(
            dataset,
            batch_size=CONFIG4RNN.PREPROCESSOR.BATCHES,
            shuffle_state=CONFIG4RNN.PREPROCESSOR.SHUFFLE,
            workers=CONFIG4RNN.PREPROCESSOR.WORKERS,
        )
        idx: int = randint(0, len(dataloader) - 1)
        # print(dataloader[idx])
        # print()

        # Load the save model parameters
        params: Path = Path(CONFIG4RNN.FILEPATHS.SAVED_NET)
        if params.exists():
            print(f"Model {params.name} Exists!")

            # Set up a model and load saved parameters
            model = NormalRNNForClassification(
                len(dictionary),
                embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
                hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
                num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
                num_classes=len(dictionary),
                dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
                accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
            )
            dict_state: dict = load(params, map_location=device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
            model.load_state_dict(dict_state)
            model.eval()
            print("Model Loaded Successfully!")

            # Predict
            TOP_K: int = 5
            total: int = 0
            correct_counter: int = 0
            with no_grad():
                for X, y in dataloader:
                    logits = model(X)
                    # Convert the output to probabilities
                    probs: Tensor = nn.functional.softmax(logits, dim=-1)
                    # print(probs.shape, probs)
                    # print()

                    # Get top k value from probabilities (values, indices)
                    _, indices = probs.topk(TOP_K, dim=-1, sorted=True)
                    # print(indices)  # size: (batches, top_k)
                    # print()

                    # Add y batch size and compare the predictions and targets
                    comparison = (y.unsqueeze(1) == indices).any(dim=-1)
                    # print(comparison)
                    # print(comparison.sum().item())

                    total += y.size(0)
                    correct_counter += comparison.sum().item()

                # Get trained model evaluation result
                starts()
                print("Trained Model Evaluation")
                lines()
                print(f"Total Samples: {total}")
                print(f"Positive Samples: {green(correct_counter)}")
                print(f"Negative Samples: {red(total - correct_counter)}")
                print(f"Top-{TOP_K} Accuracy: {correct_counter / total:.2%}")
                starts()
                print()
                """
                ****************************************************************
                Trained Model Evaluation
                ----------------------------------------------------------------
                Total Samples: 995682
                Positive Samples:       264744
                Negative Samples:       730938
                Top-5 Accuracy: 26.59%
                ****************************************************************
                """
        else:
            print(f"Model {params.name} does not exist!")


if __name__ == "__main__":
    main()
