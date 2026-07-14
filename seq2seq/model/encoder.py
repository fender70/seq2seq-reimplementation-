import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first=True
        )
        
    def forward(
        self,
        src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # type hint
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, hidden, cell
