import torch
import torch.nn as nn


class EmbeddingStock(nn.Module):
    """Embedder using simple one hot encoding."""
    def __init__(self, input_size, max_length):
        """Embedder using simple one hot encoding.

            Parameters
            ----------
            input_size : int
                Maximum number of inputs to one_hot encode
            """
        super().__init__()
        self.input_size    = input_size
        self.embedding_dim = input_size
        self.max_length = max_length

    def forward(self, batch_edge):
        """Create one-hot encoding of input

            Parameters
            ----------
            X : torch.Tensor of shape=(n_samples,)
                Input to encode.

            Returns
            -------
            result : torch.Tensor of shape=(n_samples, input_size)
                One-hot encoded version of input
            """
        concat_embed = []
        for i in range(self.max_length):
            packet = batch_edge[:, i * self.embedding_dim: self.embedding_dim * (i + 1)]
            concat_embed.append(packet)
        concat_embed = torch.hstack(concat_embed)
        return concat_embed.view(batch_edge.shape[0], self.max_length, self.embedding_dim)
