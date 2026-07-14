import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output = nn.Linear(
            in_features=hidden_dim,
            out_features=vocab_size,
        )

    def forward(
        self,
        input_token,
        hidden,
        cell
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_token).unsqueeze(1)
        outputs, (hidden, cell) = self.lstm(
            embedded,
            (hidden, cell),
        )
        outputs = outputs.squeeze(1)
        logits = self.output(outputs)

        return logits, hidden, cell
