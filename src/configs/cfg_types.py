#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/3 23:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   cfg_types.py
# @Desc     :   

from enum import StrEnum, IntEnum, unique


@unique
class SeqTaskMode(StrEnum):
    SEQ2ONE = "seq2one"
    SEQ2SEQ = "seq2seq"
    SEQ_SLICE = "slice"


@unique
class LangType(StrEnum):
    CN = "cn"
    EN = "en"


@unique
class TokenStrType(StrEnum):
    PAD = "<PAD>"
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"


@unique
class TokenIntType(IntEnum):
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3


@unique
class LSTMTaskType(StrEnum):
    CLASSIFICATION = "classification"
    GENERATION = "generation"


if __name__ == "__main__":
    out = SeqTaskMode.SEQ2ONE
    print(out)
