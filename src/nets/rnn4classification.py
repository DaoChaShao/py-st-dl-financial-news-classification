#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/4 01:48
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   rnn4classification.py
# @Desc     :   

from torch import nn, zeros, randint, device, cat

WIDTH: int = 64


class NormalRNNForClassification(nn.Module):
    """ A normal RNN model for multi-class classification tasks using PyTorch """

    def __init__(
            self,
            vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int,
            num_classes: int, dropout_rate: float = 0.3, bid: bool = True,
            accelerator: str = "cpu"
    ):
        super().__init__()
        """ Initialise the CharsRNNModel class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        :param num_classes: number of output classes
        :param dropout_rate: dropout rate for regularization
        :param bid: whether the RNN is bidirectional
        :param accelerator: accelerator for PyTorch
        """
        self._L = vocab_size  # Lexicon/Vocabulary size
        self._H = embedding_dim  # Embedding dimension
        self._M = hidden_size  # Hidden dimension
        self._C = num_layers  # RNN layers count
        self._accelerator = accelerator
        self._factor = 2 if bid else 1
        dropout = dropout_rate if self._C > 1 else 0.0

        self._embed = nn.Embedding(self._L, self._H)
        self._rnn = nn.RNN(self._H, self._M, self._C, batch_first=True, bidirectional=bid, dropout=dropout)
        self._linear = nn.Linear(self._M * self._factor, num_classes, bias=True)

        self._init_weights()

    def _init_weights(self):
        """ Initialize model parameters """
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _init_hidden(self, batch_size):
        """ Initialize h0 with correct shape and device """
        # h0 shape: (num_layers * num_directions, batch, hidden_size)
        return zeros(self._C * self._factor, batch_size, self._M, device=device(self._accelerator))

    def forward(self, X):
        """ Forward pass of the model
        :param X: input tensor, shape (batch_size, sequence_length)
        :return: output tensor and new hidden state tensor, shapes (batch_size, sequence_length, vocab_size) and (num_layers, batch_size, hidden_dim)
        """
        # X: Batch, seq_len
        # Embeddings: (batch, seq_len, embedding_dim)
        embeddings = self._embed(X)

        # Initialise h0: (num_layers * num_directions, batch, hidden_size)
        batches = X.size(0)
        h0 = self._init_hidden(batches)
        # output: (batch, seq_len, hidden_size * directions)
        # h_n:    (num_layers * directions, batch, hidden_size)
        output, h_n = self._rnn(embeddings, h0)

        # Whether it is bidirectional or not
        if self._factor == 2:
            # last layer forward = h_n[-2], backward = h_n[-1]
            last_hidden = cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]  # Or last_hidden = output[:, -1, :]

        # Linear classify: (batch, num_classes) Due to next words, num_classes = vocab_size
        out = self._linear(last_hidden)

        return out

    def summary(self):
        """ Print the model summary """
        print("=" * WIDTH)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model Summary for {self.__class__.__name__}")
        print("-" * WIDTH)
        print(f"- Vocabulary size: {self._L}")
        print(f"- Embedding dim: {self._H}")
        print(f"- Hidden size: {self._M}")
        print(f"- Num layers: {self._C}")
        print(f"- Output classes: {self._linear.out_features}")
        print(f"- Total parameters: {total_params:,}")
        print(f"- Trainable parameters: {trainable_params:,}")
        print("=" * WIDTH)
        print()


if __name__ == "__main__":
    vocab_size: int = 7459
    batch_size: int = 16
    seq_len: int = 111

    # Initialise the model
    model = NormalRNNForClassification(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        num_classes=vocab_size,  # Predict next word, num_classes = vocab_size
        dropout_rate=0.5,
        bid=True
    )
    model.summary()

    # Set up fake X
    X = randint(0, vocab_size, (batch_size, seq_len))
    output = model(X)

    print(f"Tester:")
    print(f"Input Size: {X.shape}")
    print(f"Output Size: {output.shape}")
    print()

    print(f"Layer Parameters:")
    embed_params = sum(p.numel() for p in model._embed.parameters())
    rnn_params = sum(p.numel() for p in model._rnn.parameters())
    linear_params = sum(p.numel() for p in model._linear.parameters())
    print(f"Embedding: {embed_params:,}")
    print(f"RNN: {rnn_params:,}")
    print(f"Linear: {linear_params:,}")
    print(f"Total: {embed_params + rnn_params + linear_params:,}")
