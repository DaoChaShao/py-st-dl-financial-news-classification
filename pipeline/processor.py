#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 19:28
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   processor.py
# @Desc     :   

from re import match
from torch.utils.data import Dataset
from tqdm import tqdm

from src.configs.cfg_base4dl import CONFIG4DL
from src.configs.cfg_types import SeqTaskMode
from src.datasets.seq_next_step import TorchDataset4SeqPredictionNextStep
from src.utils.helper import Timer
from src.utils.highlighter import starts, lines
from src.utils.nlp import count_frequency, build_word2id_seqs, check_vocab_coverage
from src.utils.stats import create_full_data_split, save_json
from src.utils.SQL import SQLiteIII
from src.utils.THU import cut_only


def process_data() -> tuple[Dataset, Dataset, int]:
    """ Main Function """
    with Timer("Process Data"):
        # Get the data from the database: Method I
        sqlite = SQLiteIII(CONFIG4DL.DATABASE.TABLE, CONFIG4DL.DATABASE.COL)
        sqlite.connect()
        data = sqlite.get_all_data()
        sqlite.close()
        # print(len(data))
        # print()

        # Separate the data
        sentences4train, sentences4valid, _ = create_full_data_split(data)

        # Set a dictionary
        # amount: int | None = 100
        amount: int | None = None
        items4train: list[str] = []
        if amount is None:
            for line in tqdm(sentences4train, total=len(sentences4train), desc="Tokenizing Train Data"):
                for item in cut_only(line):
                    if item in CONFIG4DL.PUNCTUATIONS.CN or match(r"^[\u4e00-\u9fff]+$", item):
                        items4train.append(item)
        else:
            for line in tqdm(sentences4train[:amount], total=amount, desc="Tokenizing Train Data"):
                for item in cut_only(line):
                    if item in CONFIG4DL.PUNCTUATIONS.CN or match(r"^[\u4e00-\u9fff]+$", item):
                        items4train.append(item)
        # print(items4train)
        tokens, _ = count_frequency(items4train, top_k=10, freq_threshold=3)
        # print(tokens)
        special: list[str] = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        dictionary: dict[str, int] = {
            word: i for i, word in
            tqdm(enumerate(special + tokens), total=len(special + tokens), desc="Building dictionary")
        }
        save_json(dictionary, CONFIG4DL.FILEPATHS.DICTIONARY)

        # Build sequences & sequence for train
        sequences4train: list[list[int]] = build_word2id_seqs(sentences4train, dictionary)
        # idx4train: int = randint(0, len(sequences4train) - 1)
        # print(sentences4train[idx4train])
        # print(sequences4train[idx4train])
        seq4train: list[int] = [
            index for line in tqdm(sequences4train, total=len(sequences4train), desc="Seq4Train") for index in line
        ]
        # print(len(seq4train))
        # Get the train dataset sentences description
        lengths: list[int] = [len(seq) for seq in sequences4train]
        max_len: int = max(lengths)
        min_len: int = min(lengths)
        avg_len: float = sum(lengths) / len(lengths)
        # Check the coverage of train data
        check_vocab_coverage(items4train, dictionary)

        # Build sequences & sequence for validation
        sequences4valid: list[list[int]] = build_word2id_seqs(sentences4valid, dictionary)
        # idx4valid: int = randint(0, len(sequences4valid) - 1)
        # print(sequences4valid[idx4valid])
        # print(sequences4valid[idx4valid])
        seq4valid: list[int] = [
            index for line in tqdm(sequences4valid, total=len(sequences4valid), desc="Seq4Valid") for index in line
        ]
        # print(len(seq4valid))
        # Check the coverage of valid data
        items4valid: list[str] = []
        if amount is None:
            for line in tqdm(sentences4valid, total=len(sentences4valid), desc="Tokenizing Valid Data"):
                for item in cut_only(line):
                    if item in CONFIG4DL.PUNCTUATIONS.CN or match(r"^[\u4e00-\u9fff]+$", item):
                        items4valid.append(item)
        else:
            for line in tqdm(sentences4valid[:amount], total=amount, desc="Tokenizing Valid Data"):
                for item in cut_only(line):
                    if item in CONFIG4DL.PUNCTUATIONS.CN or match(r"^[\u4e00-\u9fff]+$", item):
                        items4valid.append(item)
        # print(items4train)
        check_vocab_coverage(items4valid, dictionary)

        # Set dataset
        dataset4train = TorchDataset4SeqPredictionNextStep(
            seq4train,
            seq_max_len=CONFIG4DL.PREPROCESSOR.MAX_SEQUENCE_LEN,
            mode=SeqTaskMode.SEQ2ONE,
            pad_token=dictionary["<PAD>"]
        )
        dataset4valid = TorchDataset4SeqPredictionNextStep(
            seq4valid,
            seq_max_len=CONFIG4DL.PREPROCESSOR.MAX_SEQUENCE_LEN,
            mode=SeqTaskMode.SEQ2ONE,
            pad_token=dictionary["<PAD>"]
        )
        # idx4train: int = randint(0, len(dataset4train) - 1)
        # print(dataset4train[idx4train])
        # idx4valid: int = randint(0, len(dataset4valid) - 1)
        # print(dataset4valid[idx4valid])
        # print()

        starts()
        print("Data Preprocessing Summary:")
        lines()
        print(f"Train dataset: {len(dataset4train)} Samples")
        print(f"Valid dataset: {len(dataset4valid)} Samples")
        print(f"Dictionary Size: {len(dictionary)}")
        print(f"The min length of the sequence: {min_len}")
        print(f"The average length of the sequence: {avg_len:.2f}")
        print(f"The max length of the sequence: {max_len}")
        starts()
        print()
        """
        ****************************************************************
        Data Preprocessing Summary:
        ----------------------------------------------------------------
        Train dataset: 6976011 Samples
        Valid dataset: 1491563 Samples
        Dictionary Size: 7459
        The min length of the sequence: 3
        The average length of the sequence: 25.71
        The max length of the sequence: 111
        ****************************************************************
        """

        return dataset4train, dataset4valid, max_len


if __name__ == "__main__":
    process_data()
