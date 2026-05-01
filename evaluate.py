import torch
import argparse
import os
import time
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from utils.dataset import get_loader
from utils.transforms import get_transforms
from models.encoder import EncoderCNN
from models.transformer import TransformerDecoder
from models.baseline_lstm import DecoderRNN  
from models.attention_lstm import SpatialAttention
from models.attention_lstm import LSTMAttention


def generate_caption(image_path, encoder, decoder, dataset, device, max_length=20): # inference part
    transform = get_transforms('val')
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) 
    with torch.no_grad(): 
        features = encoder(image_tensor)
        start_time = time.time() # measure time
        sample_output = decoder.sample(features, max_length=max_length) # for both attention and without
        end_time = time.time()
        inference_ms = (end_time - start_time) * 1000
    if isinstance(sample_output, tuple): # api stuff
        sampled_ids, alphas = sample_output
        alphas = alphas.cpu().numpy() # Convert to numpy  matplotlib
    else:
        sampled_ids = sample_output
        alphas = None
    sampled_ids = sampled_ids[0].cpu().numpy() 
    caption_words = []
    for token_id in sampled_ids:
        word = dataset.vocab.itos[token_id]
        if word == "<START>":
            continue
        caption_words.append(word)
        if word == "<END>":
            break
    return caption_words, inference_ms, alphas

def plot_attention_heatmap(image_path, caption_words, alphas, save_path="attention_heatmap.png"): # Explain part
    image = Image.open(image_path).convert("RGB")
    num_words = len(caption_words)     # this makes a dynamic grid based on how many words were generated
    cols = 5
    rows = (num_words + cols - 1) // cols
    fig = plt.figure(figsize=(15, 3 * rows))
    for i in range(num_words):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image)
        if alphas is not None and i < len(alphas): # if attention
            alpha_grid = alphas[i].reshape(7, 7) # 49 to 7x7
            alpha_img = Image.fromarray(alpha_grid) # shape 7x7 to image
            alpha_img = alpha_img.resize(image.size, Image.Resampling.BILINEAR)
            ax.imshow(np.array(alpha_img), cmap='jet', alpha=0.5)
        ax.set_title(caption_words[i], fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"saved heatmap to{save_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Inference and Profiling on Image Captioning Models")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved .pth weights")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.checkpoint}...") 
    checkpoint = torch.load(args.checkpoint, map_location=device) 
    config = checkpoint['model_config'] 
    arch_cfg = config['architecture']
    model_type = config['model_type']
    vocab_size = checkpoint['vocab_size']
    data_dir = os.path.join(os.getcwd(), 'data')
    images_dir = os.path.join(data_dir, 'images')
    _, dataset = get_loader(
        root_dir=images_dir,
        ann_file=os.path.join(data_dir, 'annotations', 'dataset_coco.json'),
        split='val', transform=get_transforms('val'), batch_size=1
    )
    encoder = EncoderCNN(arch_cfg['embed_size']).to(device) 
    if model_type == "lstm":  
        decoder = DecoderRNN(**arch_cfg, vocab_size=vocab_size).to(device)
    elif model_type == "attention":
        decoder = LSTMAttention(
            **arch_cfg,
            vocab_size=vocab_size,
            start_token=dataset.vocab.stoi["<START>"],
            stop_token=dataset.vocab.stoi["<END>"],
            seq_length=20
        ).to(device)
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
    encoder.load_state_dict(checkpoint['encoder_state_dict'])     
    encoder.eval()
    decoder.load_state_dict(checkpoint['decoder_state_dict'])  
    decoder.eval()
    
    caption_words, latency, alphas = generate_caption(args.image, encoder, decoder, dataset, device) # runs profiler
    
    final_caption = " ".join([w for w in caption_words if w != "<END>"])
    print("\n" + "="*50)
    print(f" PREDICTED CAPTION: {final_caption}")
    print(f" INFERENCE SPEED:   {latency:.2f} ms")
    print("="*50 + "\n")
    if alphas is not None:
        plot_attention_heatmap(args.image, caption_words, alphas)
    else:
        print(" No attention weights returned by the model. Heatmap generation skipped.")

if __name__ == "__main__":
    main()