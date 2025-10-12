import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from flow_encoder.decoders import Decoder
from flow_encoder.embedding import EmbeddingStock
from flow_encoder.encoders import Encoder


class ContextBuilder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=1,
                 max_length=10, bidirectional=False, LSTM=False, device=None,
                 attention=True, rnn=True):
        """ContextBuilder that learns to interpret context from security events.
            Based on an attention-based Encoder-Decoder architecture.
            ContextBuilder可以学习从安全事件中解释上下文。基于一个基于注意力的编码器-解码器架构。
            Parameters
            ----------
            input_size : int   i.e. 30
                Size of input vocabulary, i.e. possible distinct input items
                输入词汇量的大小，即可能的不同输入项目

            output_size : int  i.e. 30
                Size of output vocabulary, i.e. possible distinct output items
                输出词汇的大小，即可能的不同的输出项目

            hidden_size : int, default=128
                Size of hidden layer in sequence to sequence prediction.
                This parameter determines the complexity of the model and its
                prediction power. However, high values will result in slower
                training and prediction times
                顺序预测中隐藏层的大小。这个参数决定了模型的复杂性和它的 预测能力。
                然而，高值会导致较慢的 训练和预测时间

            num_layers : int, default=1
                Number of recurrent layers to use

            max_length : int, default=10
                Maximum length of input sequence to expect

            bidirectional : boolean, default=False
                If True, use a bidirectional encoder and decoder

            LSTM : boolean, default=False
                If True, use an LSTM as a recurrent unit instead of GRU
            """
        super().__init__()
        self.device = device

        ################################################################
        #                      Initialise layers                       #
        ################################################################

        # Create embedding
        # [30, 128] --> [148, 256]
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embeddingStock = EmbeddingStock(input_size, max_length)
        # Create encoder
        self.encoder = Encoder(
            input_size = input_size,
            max_length = max_length,
            # embedding=self.embeddingStock,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            LSTM=LSTM,
            use_rnn=rnn,
        )

        self.decoder = Decoder(
            input_size=input_size * (bidirectional + 1) if attention else hidden_size,
            output_size=output_size,
            embedding=self.embedding,
            context_size=hidden_size,
            attention_size=max_length,
            num_layers=num_layers,
            dropout=0.1,
            bidirectional=bidirectional,
            LSTM=LSTM,
            attention_aggregate=attention,
            use_rnn=rnn,
        )

    ########################################################################
    #                        ContextBuilder Forward                        #
    ########################################################################
    def forward(self, X, is_predict=False):
        """Forwards datasets through ContextBuilder.

            Parameters
            ----------
            X : torch.Tensor of shape=(N, num_window_packets * real_packet_length)
                Tensor of input events to forward.
                要转发的输入事件的张量。
            steps : int, default=1
                Number of steps to predict in the future.
                预测未来的步骤数。
            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, steps, output_size)
                The confidence level of each output event.
                每个输出事件的置信度。

            attention : torch.Tensor of shape=(n_samples, steps, seq_len)
                Attention corresponding to X given as (batch, out_seq, in_seq).
            """

        ####################################################################
        #                           Forward datasets                           #
        ####################################################################

        # Encode input
        # x_encoded [256, 10, 30]  --> [batch_size, num_packets, packet_feature_dim]
        # context_vector [1, 256, 128] [num_layers, batch, hidden]
        X_encoded, context_vector = self.encoder(X)

        return self.decoder(X_encoded, context_vector, is_predict)

    ########################################################################
    #                         Fit/predict methods                          #
    ########################################################################
    def fit(self, X, y, epochs=10, batch_size=128, learning_rate=0.01,
            optimizer=optim.SGD, verbose=True):
        """Fit the sequence predictor with labelled datasets

            Parameters
            ----------
            X : array-like of type=int and shape=(n_samples, context_size)
                Input context to train with.

            y : array-like of type=int and shape=(n_samples, n_future_events)
                Sequences of target events.

            epochs : int, default=10
                Number of epochs to train with.

            batch_size : int, default=128
                Batch size to use for training.

            learning_rate : float, default=0.01
                Learning rate to use for training.

            optimizer : optim.Optimizer, default=torch.optim.SGD
                Optimizer to use for training.

            verbose : boolean, default=True
                If True, prints progress.

            Returns
            -------
            self : self
                Returns self
            """

        # Get current mode
        mode = self.training

        # Set to training mode
        self.train()

        # Set criterion and optimiser
        optimizer = optimizer(params=self.parameters(), lr=learning_rate)
        scaler = GradScaler()

        # Load dataset
        data_loader = DataLoader(TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32)
        ), batch_size=batch_size, shuffle=True)

        # Loop over each epoch
        for epoch in range(1, epochs + 1):
            # Set progress bar if necessary
            if verbose:
                data_loader = tqdm(data_loader,
                                   desc="[Epoch {:{width}}/{:{width}} loss={:.4f}]"
                                   .format(epoch, epochs, 0, width=len(str(epochs)))
                                   )

            try:
                # Set average loss
                total_loss, total_items = 0, 0
                # Loop over entire dataset
                for X_, y_ in data_loader:
                    # Clear gradients
                    optimizer.zero_grad()

                    with autocast():
                        # Get prediction
                        X_, y_ = X_.to(self.device), y_.to(self.device)

                        X_hat, _ = self.forward(X_)

                        # Compute loss
                        loss = torch.mean(torch.sum((y_ - X_hat) ** 2, dim=1))
                        # Back propagate
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        # loss.backward()
                        # optimizer.step()

                        # Update description
                        total_loss += loss.item()
                        total_items += X_.shape[0]

                    if verbose:
                        data_loader.set_description(
                            "[Epoch {:{width}}/{:{width}} loss={:.4f}]"
                            .format(epoch, epochs, total_loss / total_items,
                                    width=len(str(epochs))))

            except KeyboardInterrupt:
                print("\nTraining interrupted, performing clean stop")
                break

        # Reset to original mode
        self.train(mode)
        return self

    @torch.no_grad()
    def predict(self, X, batch_size):
        """Predict the next elements in sequence.

            Parameters
            ----------
            X : torch.Tensor
                input sequences

            steps : int, default=1
                Number of steps to predict into the future
                预测未来的步数

            Returns
            -------
            confidence : torch.Tensor of shape=(n_samples, seq_len, output_size)
                The confidence level of each output

            attention : torch.Tensor of shape=(n_samples, input_length)
                Attention corresponding to X given as (batch, out_seq, seq_len)
            """

        # Get current mode
        mode = self.training
        # Set to prediction mode
        self.eval()
        loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32)),
                            batch_size=batch_size, shuffle=False)
        data_list = []
        for batch in loader:
            with autocast():
                _, hidden_flow = self.forward(batch[0].to(self.device))
                data_list.append(hidden_flow.detach().cpu())
        # Reset to original mode
        self.train(mode)
        return torch.cat(data_list)

    def fit_predict(self, X, y, epochs=10, batch_size=128, learning_rate=0.01,
                    optimizer=optim.SGD, verbose=True):
        """Fit the sequence predictor with labelled datasets

            Parameters
            ----------
            X : torch.Tensor
                input sequences

            y : torch.Tensor
                output sequences

            epochs : int, default=10
                Number of epochs to train with

            batch_size : int, default=128
                Batch size to use for training

            learning_rate : float, default=0.01
                Learning rate to use for training

            optimizer : optim.Optimizer, default=torch.optim.SGD
                Optimizer to use for training

            verbose : boolean, default=True
                If True, prints progress

            Returns
            -------
            result : torch.Tensor
                Predictions corresponding to X
            """

        # Apply fit and predict in sequence
        return self.fit(
            X=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            verbose=verbose,
        ).predict(X, batch_size)
