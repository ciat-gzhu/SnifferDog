import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 input_size, max_length,
                 hidden_size, num_layers=1, embedding = None,
                 bidirectional=False, LSTM=False, use_rnn=True):
        """Encoder part for encoding sequences.

            Parameters
            ----------
            embedding : nn.Embedding
                Embedding layer to use

            hidden_size : int
                Size of hidden dimension

            num_layers : int, default=1
                Number of recurrent layers to use

            bidirectional : boolean, default=False
                If True, use bidirectional recurrent layer

            LSTM : boolean, default=False
                If True, use LSTM instead of GRU
            """
        super(Encoder, self).__init__()
        self.max_length = max_length
        self.input_size = input_size
        # self.embedding = embedding
        self.use_rnn = use_rnn
        if use_rnn:
            # Set embedding dimension
            embedding_size = self.embedding.embedding_dim if hasattr(self, 'embedding') else input_size
            self.embedding_size = embedding_size
            # Set hidden size
            self.hidden_size = hidden_size
            # Set number of layers
            self.num_layers = num_layers
            # Set bidirectional
            self.bidirectional = bidirectional
            # Set LSTM
            self.LSTM = LSTM
            self.recurrent = (nn.LSTM if LSTM else nn.GRU)(
                input_size=embedding_size,  # 148
                hidden_size=self.hidden_size,  # 256
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )

    def forward(self, input, hidden=None):
        """Encode data

            Parameters
            ----------
            input : torch.Tensor
                Tensor to use as input

            hidden : torch.Tensor
                Tensor to use as hidden input (for storing sequences)

            Returns
            -------
            output : torch.Tensor
                Output tensor

            hidden : torch.Tensor
                Hidden state to supply to next input
            """

        # hidden [1, 256, 128] [num_layers, batch, hidden]
        # input [batch, d] [512, 10]  --> [512, 592]
        # embedded [512, 10, 30] --> [512, 4, 148]
        # Get input as embedding
        embedded = input.view(-1, self.max_length, self.input_size)
        # embedded = self.embedding(input)
        if self.use_rnn:
            # Initialise hidden if not given
            if hidden is None:
                hidden = self.init_hidden(input)
            # Pass through recurrent layer
            _, hidden = self.recurrent(embedded, hidden)
            if self.bidirectional:
                embedded = torch.cat((embedded, embedded), dim=2)
        else:
            hidden = None
        return embedded, hidden

    def init_hidden(self, X):
        """Create initial hidden vector for datasets X

            Parameters
            ----------
            X : torch.Tensor,
                Tensor for which to create a hidden state

            Returns
            -------
            result : torch.Tensor
                Initial hidden tensor
            """
        if self.LSTM:
            return (torch.zeros(self.num_layers*(1 + int(self.bidirectional)),
                                X.shape[0], self.hidden_size, device=X.device),
                    torch.zeros(self.num_layers*(1 + int(self.bidirectional)),
                                X.shape[0], self.hidden_size, device=X.device))
        else:
            return torch.zeros(self.num_layers*(1 + int(self.bidirectional)),
                               X.shape[0], self.hidden_size, device=X.device)
