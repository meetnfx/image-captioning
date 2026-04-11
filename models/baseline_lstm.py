"""
To work with the jumbo i did prob should have something that follows this

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