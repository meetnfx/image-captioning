import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.1, **kwargs):
        """
        Args:
            embed_size:  Must match EncoderCNN output size (e.g. 256)
            hidden_size: LSTM hidden state size (e.g. 512)
            vocab_size:  Total number of words in vocabulary
            num_layers:  Number of LSTM layers
            dropout:     Dropout probability
        """
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Used by train.py during training (teacher forcing).

        Args:
            features: [batch_size, embed_size]         - image features from EncoderCNN
            captions: [batch_size, seq_length - 1]     - caption tokens excluding <END>

        Returns:
            predictions: [batch_size, seq_length - 1, vocab_size]
        """
        embeddings = self.dropout(self.embed(captions))   # [batch, seq_len-1, embed_size]
        features = features.unsqueeze(1)                  # [batch, 1, embed_size]
        inputs = torch.cat((features, embeddings), dim=1) # [batch, seq_len, embed_size]
        hiddens, _ = self.lstm(inputs)                    # [batch, seq_len, hidden_size]
        hiddens = self.dropout(hiddens)
        predictions = self.linear(hiddens)                # [batch, seq_len, vocab_size]
        return predictions[:, :-1, :]                     # [batch, seq_len-1, vocab_size]

    def sample(self, features, max_length=20):
        """
        Used by evaluate.py for greedy decoding on a new image.

        Args:
            features:   [1, embed_size]  - single image feature from EncoderCNN
            max_length: int              - max words to generate

        Returns:
            predicted_ids: [1, generated_length] - tensor of predicted word IDs
        """
        predicted_ids = []
        inputs = features.unsqueeze(1)  # [1, 1, embed_size]
        states = None

        for _ in range(max_length):
            hiddens, states = self.lstm(inputs, states)    # [1, 1, hidden_size]
            output = self.linear(hiddens.squeeze(1))       # [1, vocab_size]
            predicted = output.argmax(dim=1)               # [1]
            predicted_ids.append(predicted.item())

            if predicted.item() == 2:                      # 2 = <END> token
                break

            inputs = self.embed(predicted).unsqueeze(1)    # [1, 1, embed_size]

        return torch.tensor(predicted_ids).unsqueeze(0)    # [1, generated_length]
