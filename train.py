import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.dataset import get_loader
from models.encoder import EncoderCNN
# ADD DECODER IMPORTS HERE TO ACTUALLY RUN THIS
# from models.baseline_lstm import DecoderRNN
# from models.attention_lstm import AttentionDecoder
# from models.transformer import TransformerDecoder

def train():    # ARGUMENT PARSER, switch models from command line if needed
    parser = argparse.ArgumentParser(description="Train Image Captioning Models")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "attention", "transformer"], help="Which decoder to use")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device 
    print(f"Training {args.model.upper()} model on {device}")
    transform = transforms.Compose([  # data load, standard forresnet50
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_dir = os.path.join(os.getcwd(), 'data')
    train_loader, dataset = get_loader(
        root_dir=os.path.join(data_dir, 'images'),
        ann_file=os.path.join(data_dir, 'annotations', 'dataset_coco.json'),
        split='train',
        transform=transform,
        batch_size=args.batch_size
    )
    vocab_size = len(dataset.vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    encoder = EncoderCNN(args.embed_size).to(device) # init models
    if args.model == "lstm":
        # decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab_size).to(device) MODIFY!!!!!!!!!!!!!!!
        print("TODO: Initialize Baseline LSTM here")
    elif args.model == "attention":
        print("TODO: Initialize Attention Decoder here")
    elif args.model == "transformer":
        print("TODO: Initialize Transformer Decoder here")
    # LOSS AND OPTIMIZER
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # loss shuold ignore <PAD> tokens
    
    # only optimize the decoder parameters and maybe part of lin layer of encoder
    # params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    # optimizer = optim.Adam(params, lr = args.lr)

    # IMPORTANT TRAINING LOOP, MIGHT NEED TO EDIT once all decoders are ready (uncomment)    
    # for epoch in range(args.epochs):
    #     print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
    #     for idx, (imgs, captions) in tqdm(enumerate(train_loader), total = len(train_loader)):
    #         imgs = imgs.to(device)
    #         captions = captions.to(device)
    #         features = encoder(imgs) # forward pass
    #         outputs = decoder(features, captions)
    #         # outputs shape: [batch_size, seq_length,vocab_size]  # find loss
    #         # captions shape: [batch_size,seq_length]
    #         loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

    #         optimizer.zero_grad() # Backward and optimize
    #         loss.backward()
    #         optimizer.step()
    #     print(f"Loss for Epoch {epoch+1}: {loss.item():.4f}")

if __name__ == "__main__":
    train()