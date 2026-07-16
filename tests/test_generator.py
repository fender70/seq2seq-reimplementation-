import torch

from seq2seq.model.encoder import Encoder
from seq2seq.model.decoder import Decoder
from seq2seq.model.seq2seq import Seq2Seq

SRC_VOCAB_SIZE = 10
TGT_VOCAB_SIZE = 10

BATCH_SIZE = 4

SRC_LENGTH = 7
TGT_LENGTH = 6

NUM_LAYERS = 2
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

SOS_IDX = 1
EOS_IDX = 2

def test_generator():
    encoder = Encoder(
        vocab_size=SRC_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )
    
    decoder = Decoder(
        vocab_size=TGT_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        target_vocab_size=TGT_VOCAB_SIZE,
    )

    src = torch.randint(
        low=3,
        high=SRC_VOCAB_SIZE,
        size=(BATCH_SIZE, SRC_LENGTH)
    )

    generated_tokens = model.generate(src, SOS_IDX, SRC_LENGTH + 1)
    assert generated_tokens.shape == (BATCH_SIZE, SRC_LENGTH + 1)
