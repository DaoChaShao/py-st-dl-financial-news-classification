#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   prediction.py
# @Desc     :   

from pathlib import Path
from re import search
from random import randint
from streamlit import (empty, sidebar, subheader, session_state,
                       button, container, rerun, spinner, columns,
                       markdown, write, slider, selectbox,
                       caption, text_input)
from torch import load, device, Tensor, no_grad, nn, argmax

from src.configs.cfg_rnn import CONFIG4RNN
from src.configs.cfg_types import LSTMTask, Language, Tokens
from src.nets.lstm4classification import LSTMRNNForClassification
from src.utils.apis import OpenAITextCompleter
from src.utils.helper import Timer, read_yaml
from src.utils.nlp import spacy_batch_tokeniser
from src.utils.PT import item2tensor
from src.utils.SQL import SQLiteIII
from src.utils.stats import load_json, create_full_data_split

empty_messages: empty = empty()
interpreter: empty = empty()
left, mid, right = columns(3, gap="medium", vertical_alignment="center", width="stretch")
display4data_title: empty = empty()
display4data = container(border=1, width="stretch")
show4dict_title: empty = empty()
show4dict: empty = empty()

session4init: list[str] = ["dictionary", "model", "data", "contents", "labels", "news", "timer4init"]
for session in session4init:
    session_state.setdefault(session, None)
session4pick: list[str] = ["X", "y", "idx", "timer4pick"]
for session in session4pick:
    session_state.setdefault(session, None)
session4pred: list[str] = ["pred", "timer4pred"]
for session in session4pred:
    session_state.setdefault(session, None)

with sidebar:
    subheader("Test Settings")

    params: Path = Path(CONFIG4RNN.FILEPATHS.SAVED_NET)
    dic: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY)
    if params.exists() and params.is_file():
        empty_messages.warning("The model & dictionary file already exists. You can initialise model first.")

        if session_state["dictionary"] is None and session_state["model"] is None and session_state["data"] is None:
            if button("Initialise Model & Dictionary & Data", type="primary", width="stretch"):
                with Timer("Next Word Prediction") as session_state["timer4pick"]:
                    # Initialise the dictionary and convert dictionary
                    session_state["dictionary"]: dict[str, int] = load_json(dic)

                    # Initialise a model and load saved parameters
                    session_state["model"] = LSTMRNNForClassification(
                        len(session_state["dictionary"]),
                        embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
                        hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
                        num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
                        num_classes=CONFIG4RNN.PARAMETERS.CLASSES,
                        dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
                        accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
                        task=LSTMTask.CLASSIFICATION
                    )
                    dict_state: dict = load(params, map_location=device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
                    session_state["model"].load_state_dict(dict_state)

                    # Initialise the test data from sqlite database
                    table: str = "news"
                    cols: dict[str, type[int | str]] = {"label": int, "content": str}
                    with SQLiteIII(table, cols, CONFIG4RNN.FILEPATHS.SQLITE) as db:
                        session_state["data"] = db.fetch_all([col for col in cols.keys()])
                        # print(len(session_state["data"]))
                        # print()
                    # pprint(session_state["data"][:3])

                    # Separate the data
                    _, _, data = create_full_data_split(session_state["data"])

                    # Tokenise the data
                    with spinner("Tokenising the data...", show_time=True, width="stretch"):
                        amount: int | None = 100
                        # amount: int | None = None
                        session_state["contents"]: list[str] = [content for _, content in data]
                        session_state["labels"]: list[int] = [label for label, _ in data]
                        if amount is None:
                            session_state["news"]: list[list[str]] = spacy_batch_tokeniser(
                                session_state["contents"],
                                lang=Language.EN,
                                batches=CONFIG4RNN.PREPROCESSOR.BATCHES
                            )
                        else:
                            session_state["news"]: list[list[str]] = spacy_batch_tokeniser(
                                session_state["contents"][:amount],
                                lang=Language.EN,
                                batches=CONFIG4RNN.PREPROCESSOR.BATCHES
                            )
                        # print(news)
                        rerun()
        else:
            empty_messages.info(f"Initialisation completed! {session_state["timer4pick"]} Pick up a data to test.")

            show4dict_title.markdown(f"**Dictionary {len(session_state['dictionary'])}**")
            show4dict.data_editor(session_state["dictionary"], hide_index=False, disabled=True, width="stretch")

            if session_state["X"] is None and session_state["y"] is None:
                if button("Pick up a Data", type="primary", width="stretch"):
                    with Timer("Pick a piece of data") as session_state["timer4pick"]:
                        # Pick up a random sequence token converting a random sentence to sequence using dictionary
                        session_state["idx"]: int = randint(0, len(session_state["news"]) - 1)
                        sequence: list[str] = session_state["news"][session_state["idx"]]
                        seq: list[int] = [session_state["dictionary"].get(item, Tokens.UNK) for item in sequence]
                        # print(seq)

                        # Convert the token to a tensor
                        X: Tensor = item2tensor(seq, embedding=True, accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR)
                        # Add batch size
                        session_state["X"] = X.unsqueeze(0)
                        # print(session_state["X"])
                        # Get the relevant label
                        session_state["y"]: Tensor = item2tensor(
                            session_state["labels"][session_state["idx"]],
                            accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR
                        )
                        # print(session_state["y"])
                        rerun()
            else:
                empty_messages.warning(
                    f"You selected a data for prediction. {session_state['timer4pick']} You can repick if needed."
                )

                display4data_title.markdown(f"**The data you selected**")
                with display4data:
                    write(session_state["contents"][session_state["idx"]])
                    write(session_state["X"])
                    write(session_state["y"])

                if session_state["pred"] is None:
                    if button("Predict", type="primary", width="stretch"):
                        with Timer("Predict") as session_state["timer4pred"]:
                            session_state["model"].eval()
                            with no_grad():
                                logits: Tensor = session_state["model"](session_state["X"])
                                # print(logist)
                                probabilities: Tensor = nn.functional.softmax(logits)
                                # print(probabilities)
                                session_state["pred"]: Tensor = argmax(probabilities, dim=1)
                                # print(session_state["pred"])
                                rerun()

                    if button("Repick", type="secondary", width="stretch"):
                        for key in session4pick:
                            session_state[key] = None
                        rerun()
                else:
                    empty_messages.success(
                        f"Prediction Completed! {session_state["timer4pred"]} You can repredict or repick."
                    )
                    correct: bool = (session_state["pred"].item() == session_state["y"].item())
                    # print(correct)

                    with left:
                        markdown(f"**Original Label** news-{session_state["idx"]}-{session_state["y"].item()}")
                        write(
                            "Positive" if session_state["y"] == 2
                            else "Negative" if session_state["y"] == 1 else "Neutral"
                        )
                    with mid:
                        markdown(
                            f"**RNN Prediction Result** news-{int(session_state["idx"])}-{session_state["pred"].item()}"
                        )
                        rate: str = "Bingo" if correct == True else "Damn"
                        write(rate)

                    # Prompt Engineering with OpenAI API
                    # key: Path = Path(CONFIG4RNN.FILEPATHS.API_KEY)
                    # config: dict = read_yaml(key)
                    # api_key: str = config["openai"]["api_key"]

                    temperature: float = slider(
                        "Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1, disabled=True,
                        help="Controls the randomness of the model's output. Lower values make it more deterministic.",
                    )
                    top_p: float = slider(
                        "Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.1, disabled=True,
                        help="Controls the diversity of the model's output by sampling from the top p% of the probability distribution.",
                    )
                    model: str = selectbox(
                        "OpenAi Model", ["gpt-3.5-turbo", "gpt-4.1-mini", "gpt-5"], index=1, disabled=False,
                        help="Select the OpenAI model to use.",
                    )
                    match model:
                        case "gpt-3.5-turbo":
                            caption(r"Cost — Input: **\$0.50**, Output: **\$1.50** / 1M tokens")
                        case "gpt-4.1-mini":
                            caption(r"Cost — Input: **\$0.40**, Output: **\$1.60** / 1M tokens")
                        case "gpt-5":
                            caption(r"Cost — Input: **\$1.25**, Output: **\$10.00** / 1M tokens")
                        case _:
                            caption("No cost information available for this model.")
                    api_key: str = text_input(
                        "OpenAI API Key",
                        max_chars=164, type="password",
                        help="OpenAI API key for authentication",
                    )
                    caption(f"The length of API key you entered is {len(api_key)} characters.")
                    if not api_key:
                        empty_messages.error("Please enter your OpenAI API key.")
                    elif api_key and not api_key.startswith("sk-"):
                        empty_messages.error("Please enter a **VALID** OpenAI API key.")
                    elif api_key and api_key.startswith("sk-") and len(api_key) != 164:
                        empty_messages.warning("The length of OpenAI API key should be 164 characters.")
                    elif api_key and api_key.startswith("sk-") and len(api_key) == 164:
                        empty_messages.success(
                            "The OpenAI API key is valid. Please enter a story theme or a story description."
                        )

                    opener = OpenAITextCompleter(api_key, temperature=temperature, top_p=top_p)
                    role: str = "You are a professional financial expert with cross-cultural expertise."
                    rating: str = """
                        0: Neutral Review
                        1: Negative Review
                        2: Positive Review
                    """
                    prompt: str = f"""
                        Give a brief explanation in Chinese after reading the following review:
                        {session_state["contents"][session_state["idx"]]}.
                        Please analyze it, give a reason, and provide a rating (Only return number) as follows:
                        {rating}.
                    """
                    explanation = opener.client(role, prompt, model=model)
                    interpreter.write(explanation)

                    match = search(r"\b[0-2]\b", explanation)
                    pred_label: int | None = int(match.group()) if match else None
                    result = "Bingo" if (pred_label == session_state["pred"].item()) else "Damn"

                    with right:
                        markdown(f"**OpenAI Prediction Result** news-{session_state['idx']}-{pred_label}")
                        write(result)

                    if button("Repick & Repredict", type="secondary", width="stretch"):
                        for key in session4pred:
                            session_state[key] = None
                        for key in session4pick:
                            session_state[key] = None
                        rerun()
    else:
        empty_messages.error("The model & dictionary file does NOT exist.")
