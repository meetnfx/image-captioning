import torch
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from utils.transforms import get_transforms
from models.encoder import EncoderCNN
from models.transformer import TransformerDecoder
# from models.baseline_lstm import DecoderRNN  modify if needed
# from models.attention_lstm import AttentionDecoder
# from models.transformer import TransformerDecoder
# TO USE THIS IS JUST A HUMAN VERIFCATION TO MAKE SURE IT ACTUALLY WORKS the actual testing is on metrics.py
#  example use on the terminal command : python evaluate.py --image test_dog.jpg --checkpoint checkpoints/transformer_best.pth
def generate_caption(image_path, encoder, decoder, dataset, device, max_length=20): #  Generates a caption for a single image
    transform = get_transforms('val') # load and trans the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) # batch dim[1, 3, 224, 224]
    with torch.no_grad(): # get features
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features, max_length=max_length) # create Token IDs need sample from decoder
    sampled_ids = sampled_ids[0].cpu().numpy() # Extract from batch
    caption = []
    for token_id in sampled_ids:
        word = dataset.vocab.itos[token_id]
        if word == "<START>":
            continue
        if word == "<END>":
            break
        caption.append(word)
    return " ".join(caption)


class CheckpointVocabulary:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = {int(idx): token for idx, token in itos.items()}


class CheckpointDataset:
    def __init__(self, stoi, itos):
        self.vocab = CheckpointVocabulary(stoi, itos)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Models")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .pth weights")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.checkpoint}...") 
    checkpoint = torch.load(args.checkpoint, map_location=device) 
    config = checkpoint['model_config'] # take blueprint and size from checkppint
    arch_cfg = config['architecture']
    model_type = config['model_type']
    vocab_size = checkpoint['vocab_size']
    data_dir = os.path.join(os.getcwd(), 'data')
    if 'vocab_stoi' not in checkpoint or 'vocab_itos' not in checkpoint:
        raise KeyError("Checkpoint is missing vocabulary mappings needed for decoding captions.")

    dataset = CheckpointDataset(checkpoint['vocab_stoi'], checkpoint['vocab_itos'])
    encoder = EncoderCNN(arch_cfg['embed_size']).to(device) # build
    decoder = None
    if model_type == "lstm":  
        # decoder = DecoderRNN(**arch_cfg, vocab_size=vocab_size).to(device)
        pass
    elif model_type == "attention":
        raise NotImplementedError("Attention evaluation is not being modified here.")
    elif model_type == "transformer":
        decoder = TransformerDecoder(
            **arch_cfg,
            vocab_size=vocab_size,
            pad_idx=dataset.vocab.stoi["<PAD>"],
            start_idx=dataset.vocab.stoi["<START>"],
            end_idx=dataset.vocab.stoi["<END>"],
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    encoder.load_state_dict(checkpoint['encoder_state_dict'])     # load weights
    encoder.eval()
    decoder.load_state_dict(checkpoint['decoder_state_dict'])  
    decoder.eval()
    caption = generate_caption(args.image, encoder, decoder, dataset, device) 
    print(f"\nPREDICTED CAPTION: {caption}\n")

if __name__ == "__main__":
    main()