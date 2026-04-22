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

class LSTMAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, start_token, stop_token, seq_length, num_layers=1):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.seq_length = seq_length
        self.stop_token = stop_token

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear_att = nn.Linear(hidden_size, 1)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(1)


    def forward(self, x):
        output, (hidden, context) = self.lstm(x)
        raw_att = self.linear_att(output).squeeze(-1)
        att_scores = self.softmax(raw_att)
        att_output = torch.bmm(att_scores.unsqueeze(1), output).squeeze(1)
        final_output = self.linear(att_output)

        return final_output
        
    def sample(self, features, max_length=20):
        current_input = torch.full(1, self.start_token)
        predicted_ids = []

        for _ in range(max_length):
            output, hidden = self.forward(current_input)
            word = output.argmax()
            predicted_ids.append(word)
            current_input = output

            if word == self.stop_token:
                break

        return predicted_ids
