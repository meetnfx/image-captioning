import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import get_loader
from models.encoder import EncoderCNN
from utils.transforms import get_transforms
# ADD DECODER IMPORTS HERE TO ACTUALLY RUN THIS
# from models.baseline_lstm import DecoderRNN
# from models.attention_lstm import AttentionDecoder
# from models.transformer import TransformerDecoder



def train():    # ARGUMENT PARSER, switch models from command line if needed EXAMPLE python train.py --model transformer
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
    data_dir = os.path.join(os.getcwd(), 'data')
    train_loader, dataset = get_loader(
        root_dir=os.path.join(data_dir, 'images'),
        ann_file=os.path.join(data_dir, 'annotations', 'dataset_coco.json'),
        split='train',
        transform=get_transforms('train'),
        batch_size=args.batch_size
    )
    val_loader, _ = get_loader(  # validation loader
        root_dir=os.path.join(data_dir, 'images'),
        ann_file=os.path.join(data_dir, 'annotations', 'dataset_coco.json'),
        split='val',
        transform=get_transforms('val'),
        batch_size=args.batch_size,
        shuffle=False # Don't need to shuffle validation data
    )
    vocab_size = len(dataset.vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    encoder = EncoderCNN(args.embed_size).to(device) # init models
    if args.model == "lstm":
        # decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab_size).to(device) MODIFY!!!!!!!!!!!!!!!
        print("TODO: Initialize Baseline LSTM here")
    elif args.model == "attention":
        # decoder MODIFY
        print("TODO: Initialize Attention Decoder here")
    elif args.model == "transformer":
        # decoder MODIFY
        print("TODO: Initialize Transformer Decoder here")

    # LOSS AND OPTIMIZER
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # loss shuold ignore <PAD> tokens
    
    # only optimize the decoder parameters and maybe part of lin layer of encoder
    # this is commented, can uncomment once models are completed and rdy for testing
    """
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    writer = SummaryWriter(f"runs/{args.model}_experiment") # create tensor board writer
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf') 
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        encoder.train() #training
        decoder.train()
        train_loss = 0
        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            imgs, captions = imgs.to(device), captions.to(device)
            features = encoder(imgs)
            # input to decoder has everything but the <END> token
            # model sees <START> and all words up to the last actual word
            outputs = decoder(features, captions[:, :-1]) 
            # Targets for Loss- Everything EXCEPT the <START> token
            # so when model see <START> it pred Word 1
            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        encoder.eval() # validation part
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, (imgs, captions) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating"):
                imgs, captions = imgs.to(device), captions.to(device)
                features = encoder(imgs)
                outputs = decoder(features, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        writer.add_scalar('Loss/Train', avg_train_loss, epoch) # write to tensor board
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        checkpoint = { # checkpoints with val loss
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'vocab_size': vocab_size
        }
        torch.save(checkpoint, os.path.join("checkpoints", f"{args.model}_latest.pth"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join("checkpoints", f"{args.model}_best.pth"))
            print(f"new best validation loss Saved checkpoint.")
    writer.close()
    """
if __name__ == "__main__":
    train()