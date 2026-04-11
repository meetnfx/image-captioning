





def forward(self, features, captions):  
        """
        REQUIRED SHAPES:
        Input:
            features: [batch_size, embed_size] (from Encoder)
            captions: [batch_size, seq_length - 1] (target tokens, shifted by train.py)
        Output:
            predictions: [batch_size, seq_length - 1, vocab_size]
        """
        # logic here
        pass