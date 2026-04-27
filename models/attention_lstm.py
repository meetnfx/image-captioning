import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: [batch, 49, encoder_dim]
        # decoder_hidden: [batch, decoder_dim]
        att1 = self.encoder_att(encoder_out) # [batch, 49, attention_dim]
        att2 = self.decoder_att(decoder_hidden) # [batch, attention_dim]
        combined = self.relu(att1 + att2.unsqueeze(1)) # [batch, 49, attention_dim]
        attention = self.full_att(combined).squeeze(2) # [batch, 49]
        alpha = self.softmax(attention) # [batch, 49]
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # [batch, encoder_dim]
        return context, alpha

class LSTMAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, start_token, stop_token, seq_length, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.stop_token = stop_token
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = SpatialAttention(embed_size, hidden_size, 256)
        self.lstm = nn.LSTMCell(embed_size + embed_size, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)
        self.f_beta = nn.Linear(hidden_size, embed_size)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out):
        # encoder_out shape: [batch, 49, embed_size]
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, features, captions):
        # features: [batch, 49, embed_size]
        # captions: [batch, seq_len]
        batch_size = features.size(0)
        seq_len = captions.size(1)
        
        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(features)
        
        predictions = torch.zeros(batch_size, seq_len, self.vocab_size).to(features.device)
        
        for t in range(seq_len):
            context, _ = self.attention(features, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context
            
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            lstm_input = self.dropout_layer(lstm_input)
            h, c = self.lstm(lstm_input, (h, c))
            
            preds = self.fc(h)
            predictions[:, t, :] = preds
            
        return predictions

    def sample(self, features, max_length=20):
        # features: [1, 49, embed_size]
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)
        device = features.device
        
        curr_word = torch.LongTensor([self.start_token]).to(device)
        predicted_ids = []
        
        for _ in range(max_length):
            embed = self.embedding(curr_word)
            context, _ = self.attention(features, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context
            
            lstm_input = torch.cat([embed, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            
            output = self.fc(h)
            word = output.argmax(1)
            predicted_ids.append(word.item())
            
            if word.item() == self.stop_token:
                break
                
            curr_word = word
            
        return torch.tensor(predicted_ids).unsqueeze(0)
