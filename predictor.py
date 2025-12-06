#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:10
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from pathlib import Path
from random import randint
from re import match
from torch import Tensor, load, device, no_grad, nn
from tqdm import tqdm

from src.configs.cfg_rnn import CONFIG4RNN
from src.nets.rnn4classification import NormalRNNForClassification
from src.utils.helper import Timer
from src.utils.PT import item2tensor
from src.utils.SQL import SQLiteIII
from src.utils.stats import create_full_data_split, load_json
from src.utils.THU import cut_only


def main() -> None:
    """ Main Function """
    # Get the data from the database: Method II
    with SQLiteIII(CONFIG4RNN.DATABASE.TABLE, CONFIG4RNN.DATABASE.COL) as db:
        data = db.get_all_data()
        print(len(data))
        print()

    with Timer("Next Word Prediction"):
        # Separate the data
        _, _, sentences = create_full_data_split(data)

        # Tokenise the data
        # amount: int | None = 100
        amount: int | None = None
        items: list[str] = []
        if amount is None:
            for line in tqdm(sentences, total=len(sentences), desc="Tokenizing Test Data"):
                for item in cut_only(line):
                    if item in CONFIG4RNN.PUNCTUATIONS.CN or match(r"^[\u4e00-\u9fff]+$", item):
                        items.append(item)
        else:
            for line in tqdm(sentences[:amount], total=amount, desc="Tokenizing Test Data"):
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

        # Pick up a random sequence token
        idx: int = randint(0, len(sequence) - 1 - CONFIG4RNN.PREPROCESSOR.MAX_SEQUENCE_LEN)
        seq_token: list[int] = sequence[idx: idx + CONFIG4RNN.PREPROCESSOR.MAX_SEQUENCE_LEN]
        # print(seq_token)
        print()

        # Convert the token to a tensor
        X: Tensor = item2tensor(seq_token, embed=True, accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR)
        # Add batch size
        X = X.unsqueeze(0)
        # print(X)
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

            # Prediction
            TOP_K: int = 5
            with no_grad():
                predictions = model(X)
                # print(predictions.shape, predictions)
                # print()

                # Set a temperature scaler controlling categorical randomness, lower values, lower randomness
                temperature: float = CONFIG4RNN.PARAMETERS.TEMPERATURE
                if 0.5 <= temperature <= 1.5:
                    predictions = predictions / temperature
                else:
                    raise ValueError("Temperature should be between 0.5 and 1.5.")

                # Convert the output to probabilities
                probs: Tensor = nn.functional.softmax(predictions, dim=-1)
                # print(probs.shape, probs)
                # print()

                # Get top k value from probabilities (values, indices)
                _, indices = probs.topk(TOP_K, dim=-1, sorted=True)
                # print(indices)  # size: (batches, top_k)
                # print()

                # Get the relevant words
                print("".join(items[idx: idx + CONFIG4RNN.PREPROCESSOR.MAX_SEQUENCE_LEN]))
                pred_words: list[str] = [reversed_dictionary.get(idx) for idx in indices.squeeze().tolist()]
                print(pred_words)
                print()
        else:
            print(f"Model {params.name} does not exist!")


if __name__ == "__main__":
    main()
