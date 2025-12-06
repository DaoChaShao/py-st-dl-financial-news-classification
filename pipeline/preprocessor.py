#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 00:07
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preprocessor.py
# @Desc     :   

from pathlib import Path
from pprint import pprint
from random import randint
from tqdm import tqdm

from src.configs.cfg_base import CONFIG
from src.utils.helper import Timer
from src.utils.SQL import SQLiteIII
from src.utils.stats import load_json


def preprocess_data() -> None:
    """ Main Function """
    with Timer("Preprocess Data"):
        path: Path = Path(CONFIG.FILEPATHS.DATA4ALL)

        if path.exists():
            print(f"Bingo! {path.name} exists!")
            print()

            # Get raw structural data
            raw: dict = load_json(path)
            idx_raw: int = randint(0, len(raw) - 1)
            pprint(raw[idx_raw])
            print(type(raw[idx_raw]), len(raw))
            print()

            # Get the specific data from the raw data
            dialogs: list[list[str]] = [line["dialog"] for line in tqdm(raw, total=len(raw), desc="Get Dialogs")]
            idx_rd: int = randint(0, len(dialogs) - 1)
            print(dialogs[idx_rd])
            print(len(dialogs), len(dialogs[idx_rd]))
            print()

            # Get the dialog contents
            data: list = [
                line.split("：", 1)[1]
                for lines in tqdm(dialogs, total=len(dialogs), desc="Get Dialog Contents")
                for line in lines
            ]
            idx: int = randint(0, len(data) - 1)
            print(data[idx])
            print(len(data), len(data[idx]))

            # Store the preprocessed data into sqlit 3 database
            sqlite = SQLiteIII(CONFIG.DATABASE.TABLE, CONFIG.DATABASE.COL)
            sqlite.insert(data)
        else:
            print(f"{path.name} does NOT exist!")


if __name__ == "__main__":
    preprocess_data()
