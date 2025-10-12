import torch
import torch.nn.functional as F
from torch import nn


class DecoderAttention(nn.Module):
    def __init__(self, embedding, context_size, attention_size, num_layers=1,
                 dropout=0.1, bidirectional=False, LSTM=False, attention_aggregate: bool = True):
        """Attention decoder for retrieving attention from context vector.

            Parameters
            ----------
            embedding : nn.Embedding
                Embedding layer to use.

            context_size : int
                Size of context to expect as input.

            attention_size : int
                Size of attention vector.

            num_layers : int, default=1
                Number of recurrent layers to use.

            dropout : float, default=0.1
                Default dropout rate to use.

            bidirectional : boolean, default=False
                If True, use bidirectional recurrent layer.

            LSTM : boolean, default=False
                If True, use LSTM instead of GRU.
            """
        # Call super
        super().__init__()

        ################################################################
        #                      Initialise layers                       #
        ################################################################
        # Embedding layer
        # [30, 128] [input, hidden]
        self.embedding = embedding
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Recurrency layer
        self.recurrent = (nn.LSTM if LSTM else nn.GRU)(
            input_size=embedding.embedding_dim,  # 128
            hidden_size=context_size,  # 128
            num_layers=num_layers,  # 1
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Attention layer [attention_size, num_window_packets-1]
        self.attention_aggregate = attention_aggregate
        if attention_aggregate:
            self.attn = nn.Linear(
                in_features=context_size * num_layers * (1 + bidirectional),
                out_features=attention_size
            )
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(attention_size)

    def forward(self, context_vector, previous_input=None):
        """Compute attention based on input and hidden state.
        根据输入和隐藏状态来计算注意力。

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, embedding_dim)
                Input from which to compute attention
                用来计算注意力的输入

            hidden : torch.Tensor of shape=(n_samples, hidden_size)
                Context vector from which to compute attention
                用于计算注意力的语境向量

            Returns
            -------
            attention : torch.Tensor of shape=(n_samples, context_size)
                Computed attention

            context_vector : torch.Tensor of shape=(n_samples, hidden_size)
                Updated context vector
            """
        # Get embedding from input: [256, 1] -> [256, 1, 128]
        embedded = self.embedding(previous_input).view(-1, 1, self.embedding.embedding_dim)
        # Apply dropout layer
        embedded = self.dropout(embedded)

        # Compute attention and pass through hidden to next state
        # embedded [N, 1, hidden_dim]
        # attention(flow_hidden) [N, 1, hidden_dim]
        # context_vector [1, N, hidden_dim]
        flow_hidden, context_vector = self.recurrent(embedded, context_vector)
        flow_hidden = flow_hidden.squeeze(1)
        if self.attention_aggregate:
            # Compute attention, Normalise attention weights, i.e. sum to 1
            # Attention Dim: [N, num_window_packets-1]
            attention = F.softmax(self.attn(flow_hidden), dim=1)
        else:
            attention = flow_hidden
        return attention, context_vector, flow_hidden


class DecoderEvent(nn.Module):
    def __init__(self, input_size, output_size, attention_aggregate):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size)
        )
        self.attention_aggregate = attention_aggregate

    def forward(self, X, attention):
        """Decode X with given attention.
        用所给的注意力解码X。

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples, context_size, hidden_size)
                Input samples on which to apply attention.
                对其进行关注的输入样本。

            attention : torch.Tensor of shape=(n_samples, context_size)
                Attention to use for decoding step
                注意用于解码步骤

            Returns
            -------
            output : torch.Tensor of shape=(n_samples, output_size)
                Decoded output
            """
        # X [256, 10, 30]
        # attention [256, 10] attention.unsqueeze(1) [256, 1, 10]
        # Apply attention (by computing batch matrix-matrix product)
        # [256, 1, 30] -> [256, 30]
        if self.attention_aggregate:
            X = torch.bmm(attention.unsqueeze(1), X).squeeze(1)
        else:
            X = attention
        return self.out(X)


class Decoder(nn.Module):
    def __init__(self, embedding, input_size, output_size, context_size, attention_size,
                 num_layers=1, dropout=0.1, bidirectional=False,
                 LSTM=False, attention_aggregate: bool = True, use_rnn: bool = True):
        super().__init__()
        self.embedding = embedding
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Attention layer [attention_size, num_window_packets-1]
        self.attention_aggregate = attention_aggregate
        if attention_aggregate:
            self.attn = nn.Linear(
                in_features=context_size * num_layers * (1 + bidirectional),
                out_features=attention_size
            )

        self.use_rnn = use_rnn
        if use_rnn:
            # Recurrency layer
            self.recurrent = (nn.LSTM if LSTM else nn.GRU)(
                input_size=embedding.embedding_dim,  # 128
                hidden_size=context_size,  # 128
                num_layers=num_layers,  # 1
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(embedding.embedding_dim, context_size),
                nn.Tanh()
            )

        self.out = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size)
        )

    def forward(self, X_encoded, context_vector=None, is_predict: bool = False):
        previous_input = torch.zeros(
            size=(X_encoded.shape[0], 1),
            dtype=torch.long,
            device=X_encoded.device,
        )
        # Get embedding from input: [256, 1] -> [256, 1, 128]
        embedded = self.embedding(previous_input).view(-1, 1, self.embedding.embedding_dim)
        # Apply dropout layer
        embedded = self.dropout(embedded)

        if self.use_rnn:
            # Compute attention and pass through hidden to next state
            # embedded [N, 1, hidden_dim]
            # attention(flow_hidden) [N, 1, hidden_dim]
            # context_vector [1, N, hidden_dim]
            flow_hidden = self.recurrent(embedded, context_vector)[0].squeeze(1)
        else:
            flow_hidden = self.projection(embedded).mean(dim=1).squeeze(1)

        if is_predict:
            out = None
        else:
            if self.attention_aggregate:
                # Compute attention, Normalise attention weights, i.e. sum to 1
                # Attention Dim: [N, num_window_packets-1]
                attention = F.softmax(self.attn(flow_hidden), dim=-1)
                X = torch.bmm(attention.unsqueeze(1), X_encoded).squeeze(1)
            else:
                X = flow_hidden

            out = self.out(X)
        return out, flow_hidden
