"""
In addition to whatever you work on for the models, To work with the jumbo i did prob should have the forward have something that follows this and a sample func

1 forward(self, features, captions)
   Used by train.py for training.
   Inputs:
       features: [batch_size, embed_size] (from Encoder)
       captions: [batch_size, seq_length - 1] (Target tokens, shifted by train.py)
   Outputs:
       predictions: [batch_size, seq_length - 1, vocab_size]

2 sample(self, features, max_length=20)
   Used by evaluate.py to generate text for a brand new image.
   Inputs:
       features: [1, embed_size] (Single image feature vector)
       max_length: int (Maximum words to generate)
   Outputs:
       predicted_ids: [1, generated_length] (tensor of the integer word IDs)
"""
import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_size,
        vocab_size,
        num_heads=8,
        num_layers=6,
        hidden_size=512,
        dropout=0.1,
        max_length=50,
        pad_idx=0,
        start_idx=1,
        end_idx=2,
        **kwargs,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.word_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.feature_projection = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def _build_causal_mask(self, seq_length, device):
        return torch.triu(
            torch.ones(seq_length, seq_length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, features, captions):
        batch_size, seq_length = captions.shape
        positions = torch.arange(seq_length, device=captions.device).unsqueeze(0).expand(batch_size, -1)

        tgt = self.word_embedding(captions) + self.position_embedding(positions)
        tgt = self.dropout(tgt)

        memory = self.feature_projection(features).unsqueeze(1)
        tgt_mask = self._build_causal_mask(seq_length, captions.device)
        tgt_padding_mask = captions.eq(self.pad_idx)

        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.output_layer(decoded)

    def sample(self, features, max_length=20):
        generated = torch.full(
            (features.size(0), 1),
            fill_value=self.start_idx,
            dtype=torch.long,
            device=features.device,
        )

        for _ in range(max_length):
            logits = self.forward(features, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if torch.all(next_token.squeeze(1) == self.end_idx):
                break

        return generated