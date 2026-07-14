import torch

from seq2seq.model.encoder import Encoder

def test_encoder_shapes():
    vocab_size = 100
    embedding_dim = 32
    hidden_dim = 64
    num_layers = 2
    batch_size = 4
    source_length = 7

    encoder = Encoder(
        vocab_size = vocab_size,
        embedding_dim = embedding_dim,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
    )

    src = torch.randint(
        low = 0,
        high = vocab_size,
        size = (batch_size, source_length),
    )

    outputs, hidden, cell = encoder(src)

    assert outputs.shape == (batch_size, source_length, hidden_dim)
    assert hidden.shape == (num_layers, batch_size, hidden_dim)
    assert cell.shape == (num_layers, batch_size, hidden_dim)
