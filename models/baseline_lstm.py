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
        self.hidden_size = kwargs.get('hidden_size', hidden_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.init_h = nn.Linear(embed_size, self.hidden_size) # map img feat to lstm first state
        self.init_c = nn.Linear(embed_size, self.hidden_size)
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
        features = features.mean(dim=1)
        h0 = self.init_h(features).unsqueeze(0) # 1, batch_size, hidden_size
        c0 = self.init_c(features).unsqueeze(0)
        embeddings = self.dropout(self.embed(captions)) # batch_size, seq_len, embed_size
        hiddens, _ = self.lstm(embeddings, (h0, c0))
        hiddens = self.dropout(hiddens)
        predictions = self.linear(hiddens)                
        return predictions


    def sample(self, features, max_length=20):
        """
        Used by evaluate.py for greedy decoding on a new image.


        Args:
            features:   [1, embed_size]  - single image feature from EncoderCNN
            max_length: int              - max words to generate


        Returns:
            predicted_ids: [1, generated_length] - tensor of predicted word IDs
        """
        features = features.mean(dim=1)
        h0 = self.init_h(features).unsqueeze(0)
        c0 = self.init_c(features).unsqueeze(0)
        states = (h0, c0)
        predicted_ids = []
        inputs = torch.tensor([[1]]).to(features.device)
        for _ in range(max_length):
            embeddings = self.embed(inputs) # 1, 1, embed_size
            hiddens, states = self.lstm(embeddings, states)     # memory based on prev word/state
            output = self.linear(hiddens.squeeze(1))      
            predicted = output.argmax(dim=1)              
            predicted_ids.append(predicted.item())
            if predicted.item() == 2: # end token
                break


            inputs = predicted.unsqueeze(1)    # pred as next word


        return torch.tensor(predicted_ids).unsqueeze(0)   # 1 generated_length

