import torch
import torch.nn as nn

from seq2seq.model.encoder import Encoder
from seq2seq.model.decoder import Decoder

class Seq2Seq(nn.Module):
    
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        target_vocab_size: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,    
    ):
        
        batch_size = src.shape[0]
        target_length = tgt.shape[1]

        _, hidden, cell = self.encoder(src)

        # Start with the first target token as the first input which is <SOS>
        input_token = tgt[:, 0]

        all_logits = torch.zeros(
            batch_size,
            tgt.shape[1] - 1,
            self.target_vocab_size,
            device=src.device
        )

        # First time step t=1, input token is still at index 0 for <SOS>
        for t in range(1, target_length):
            logits, hidden, cell = self.decoder(
                input_token,
                hidden,
                cell
            )
            
            all_logits[:, t - 1, :] = logits
            
            input_token = tgt[:, t]

        # Return all logits for each time position, for each sequence in the batch
        return all_logits
        
    # TODO(Cedric): Implement early stopping if generate EOS token
    def generate(
        self,
        src: torch.Tensor,
        sos_idx: int,
        max_length: int,
    ) -> torch.Tensor:
        
        batch_size = src.shape[0]

        input_token = torch.full(
            (batch_size,),
            sos_idx,
            dtype=torch.long,
            device=src.device
        )

        _, hidden, cell = self.encoder(src)

        generated_steps = []

        for _ in range(max_length):
            logits, hidden, cell = self.decoder(
                input_token,
                hidden,
                cell,
            )

            input_token = logits.argmax(dim=-1)
            generated_steps.append(input_token)

        return torch.stack(generated_steps, dim=1)

