import torch
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from utils.dataset import get_loader
from utils.transforms import get_transforms
from models.encoder import EncoderCNN
# from models.baseline_lstm import DecoderRNN  modify if needed
# from models.attention_lstm import AttentionDecoder
# from models.transformer import TransformerDecoder


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

def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Models")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "attention", "transformer"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .pth weights")
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(os.getcwd(), 'data') # load vocab dict
    _, dataset = get_loader(
        root_dir=os.path.join(data_dir, 'images'),
        ann_file=os.path.join(data_dir, 'annotations', 'dataset_coco.json'),
        split='val', transform=get_transforms('val'), batch_size=1
    )
    vocab_size = len(dataset.vocab)
    encoder = EncoderCNN(args.embed_size).to(device) # init models
    decoder = None
    if args.model == "lstm":  # MODIFY THIS ADD MODELS
        # decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab_size).to(device)
        pass






    print(f"Loading weights from {args.checkpoint}...") 
    checkpoint = torch.load(args.checkpoint, map_location=device) 
    encoder.load_state_dict(checkpoint['encoder_state_dict'])    
    encoder.eval()
    # UNCOMMENT THE LINES BELOW ONCE YOUR DECODER IS RDY
    # decoder.load_state_dict(checkpoint['decoder_state_dict'])  
    # decoder.eval()
    # caption = generate_caption(args.image, encoder, decoder, dataset, device) 
    # print(f"\n PREDICTED CAPTION: {caption}\n")  

if __name__ == "__main__":
    main()