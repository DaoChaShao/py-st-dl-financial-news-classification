#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, expander, caption, empty, divider

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("Financial News Prediction")
with expander("**INTRODUCTION**", expanded=True):
    caption("+ 基于 Streamlit 的交互式金融新闻情感分类应用，支持 LSTM-RNN 模型实时推断。")
    caption("+ 集成模型加载、字典映射、序列构建与文本向量化等核心 NLP 处理流程。")
    caption("+ 支持从 SQLite 数据库读取新闻内容与标签，并自动进行数据分割与随机抽样。")
    caption("+ 提供基于 spaCy 的批量分词功能，实现高效的文本预处理与序列生成。")
    caption("+ 支持模型推理，包括 logits 计算、Softmax 概率获取与 argmax 分类输出。")
    caption("+ 可视化展示原文、真实标签、模型预测结果与正确性判断。")
    caption("+ 集成 OpenAI 推理接口，可生成中文解释并返回情感评分，与本地模型结果进行对比。")
    caption("+ 提供计时器显示，包括初始化时间、抽样时间、推理时间，为性能评估提供参考。")
    caption("+ 支持一键重新抽样与重新预测，提供完整的可重复实验流程。")
    caption("+ 适用于教学展示、模型验证与 NLP 情感分类任务的实验平台。")
    divider()
    caption("+ Streamlit-based interactive system for evaluating an LSTM financial-news classification model.")
    caption("+ Supports model initialisation, dictionary loading, and SQLite-backed dataset retrieval.")
    caption("+ Provides automatic random sampling of news items with tokenisation via spaCy batch processing.")
    caption("+ Converts tokens to indexed sequences using a persistent dictionary with UNK fallback handling.")
    caption("+ Offers real-time prediction using a trained LSTM classifier with softmax probability estimation.")
    caption("+ Displays original labels, predicted labels, correctness checks, and detailed Streamlit UI feedback.")
    caption("+ Integrates OpenAI API for generating expert-style Chinese explanations and external rating predictions.")
    caption("+ Automatically extracts numerical ratings from LLM outputs for cross-model comparison and validation.")
    caption("+ Presents a full workflow: data selection → preprocessing → tensor construction → inference → evaluation.")
    caption("+ Designed as a modular, educational, and debugging-friendly tool for text classification experiments.")
