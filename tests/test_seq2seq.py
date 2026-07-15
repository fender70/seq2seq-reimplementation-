import torch

from seq2seq.model.seq2seq import Seq2Seq
from seq2seq.model.encoder import Encoder
from seq2seq.model.decoder import Decoder

SRC_VOCAB_SIZE = 100
TGT_VOCAB_SIZE = 120

BATCH_SIZE = 4

SRC_LENGTH = 7
TGT_LENGTH = 6

NUM_LAYERS = 2
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

SOS_IDX = 1

def test_seq2seq():
    encoder = Encoder(
        vocab_size=SRC_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    )

    decoder = Decoder(
        vocab_size=TGT_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    )

    src = torch.randint(
        low=0,
        high=SRC_VOCAB_SIZE,
        size=(BATCH_SIZE, SRC_LENGTH)
    )

    tgt = torch.randint(
        low=0,
        high=TGT_VOCAB_SIZE,
        size=(BATCH_SIZE, TGT_LENGTH)
    )

    tgt[:, 0] = SOS_IDX

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        target_vocab_size=TGT_VOCAB_SIZE,
    )

    all_logits = model(src, tgt)

    assert all_logits.shape == (BATCH_SIZE, TGT_LENGTH - 1, TGT_VOCAB_SIZE)
