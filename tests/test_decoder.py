import torch

from seq2seq.model.encoder import Encoder
from seq2seq.model.decoder import Decoder

V_SRC_SIZE = 100
V_TGT_SIZE = 100

EMBEDDING_DIM = 32
HIDDEN_DIM = 64
LAYERS = 2
BATCH_SIZE = 4
SOURCE_LENGTH = 7

def test_encoder_decoder_pipeline():
    
    encoder = Encoder(
            vocab_size=V_SRC_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=LAYERS
            )
    decoder = Decoder(
            vocab_size=V_TGT_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=LAYERS,
            )

    src = torch.randint(
            low=0,
            high=V_SRC_SIZE,
            size=(BATCH_SIZE, SOURCE_LENGTH)
            )

    encoded_outputs, encoded_hidden, encoded_cell = encoder(src)

    SOS_IDX = 1

    input_token = torch.full(
            size=(BATCH_SIZE,),
            fill_value=SOS_IDX,
            dtype=torch.long,
            )

    logits, hidden, cell = decoder(input_token, encoded_hidden, encoded_cell)

    predicted_token = logits.argmax(dim=1)

    assert src.shape == (BATCH_SIZE, SOURCE_LENGTH)
    assert encoded_outputs.shape == (BATCH_SIZE, SOURCE_LENGTH, HIDDEN_DIM)
    assert encoded_hidden.shape == (LAYERS, BATCH_SIZE, HIDDEN_DIM)
    assert encoded_cell.shape == (LAYERS, BATCH_SIZE, HIDDEN_DIM)
    assert input_token.shape == (BATCH_SIZE,)
    assert logits.shape == (BATCH_SIZE, V_TGT_SIZE)
    assert hidden.shape == (LAYERS, BATCH_SIZE, HIDDEN_DIM)
    assert cell.shape == (LAYERS, BATCH_SIZE, HIDDEN_DIM)
    assert predicted_token.shape == (BATCH_SIZE,)

    print(predicted_token)
