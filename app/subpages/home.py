#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, expander, caption, empty

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("Next Word Prediction Application")
with expander("**INTRODUCTION**", expanded=True):
    caption("+ Streamlit-based interactive application for demonstrating an RNN-driven next-word prediction model trained on Chinese corpora.")
    caption("+ Integrates model loading, vocabulary decoding, and dynamic token-to-text mapping utilities.")
    caption("+ Supports SQLite-backed dataset retrieval with random sampling for real-time inference cases.")
    caption("+ Provides tokenisation, sequence generation, temperature scaling, and configurable prediction workflows.")
    caption("+ Enables top-K next-token prediction with visualised probability outputs.")
    caption("+ Offers an end-to-end workflow including model initialisation, text preprocessing, sequence construction, inference execution, and result inspection.")
    caption("+ Designed as a practical teaching and experimentation tool for sequence modelling, deep learning inference, and data processing.")

