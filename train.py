import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import torchvision.transforms as transforms
from tqdm import tqdm
import yaml
from utils.dataset import get_loader
from models.encoder import EncoderCNN
from utils.transforms import get_transforms
from models.attention_lstm import AttentionDecoder
import matplotlib.pyplot as plt

def train():
    parser = argparse.ArgumentParser(description="Train Image Captioning Models")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_type = config['model_type']
    train_cfg = config['training']
    arch_cfg = config['architecture']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_type.upper()} model on {device}")

    data_dir = os.path.join(os.getcwd(), "data")
    images_dir = os.path.join(data_dir,  "images")

    train_loader, dataset = get_loader(
        root_dir=images_dir,
        ann_file=os.path.join(data_dir, 'annotations', 'dataset_coco.json'),
        split='train',
        transform=get_transforms('train'),
        batch_size=train_cfg['batch_size'] 
    )
    val_loader, _ = get_loader(
        root_dir=images_dir,
        ann_file=os.path.join(data_dir, 'annotations', 'dataset_coco.json'),
        split='val',
        transform=get_transforms('val'),
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        vocab=dataset.vocab
    )

    vocab_size = len(dataset.vocab)

    pad_token = dataset.vocab.stoi["<PAD>"]

    encoder = EncoderCNN(arch_cfg["embed_size"]).to(device)

    if model_type == "lstm":
        pass
        # decoder = DecoderRNN(**arch_cfg, vocab_size=vocab_size).to(device)
        # print("TODO: Initialize Baseline LSTM here")
    elif model_type == "attention":
        decoder = AttentionDecoder(
            **arch_cfg,
            vocab_size=vocab_size,
            start_token=dataset.vocab.stoi["<START>"],
            stop_token=dataset.vocab.stoi["<END>"],
            seq_length=20
        ).to(device)
    elif model_type == "transformer":
        pass
        # decoder = TransformerDecoder(**arch_cfg, vocab_size=vocab_size).to(device)
        # print("TODO: Initialize Transformer Decoder here")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)


    params = list(decoder.parameters()) + list(encoder.projection.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0))

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')

    avg_train_loss_history = []
    avg_validation_loss_history = []

    for epoch in range(train_cfg["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{train_cfg["epochs"]} ---")
        encoder.train()
        decoder.train()
        train_loss = 0

        for _, (images, captions) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            images = images.to(device)
            captions = captions.to(device)
            features = encoder(images)

            outputs = decoder(features, captions[:, :-1])

            loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        avg_train_loss_history.append(avg_train_loss)

        encoder.eval()
        decoder.eval()

        val_loss = 0
        with torch.no_grad():
            for _, (images, captions) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating"):
                images, captions = images.to(device), captions.to(device) 
                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        avg_validation_loss_history.append(avg_val_loss)

        checkpoint = {
            "epoch": epoch + 1,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": vocab_size,
            "model_config": config
        }

        torch.save(checkpoint, os.path.join("checkpoints", f"{model_type}_latest.pth"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join("checkpoints", f"{model_type}_best.pth"))
            print(f"New best validation loss! Saved checkpoint.")

    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_loss_history, '-o', label="Training Loss")
    plt.plot(avg_validation_loss_history, '-o', label="Validation Loss")
    plt.title(f"Training vs Validation Loss ({model_type})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("train_val_loss.png")
    print("Training/Validation Loss chart saved to train_val_loss.png")
            

if __name__ == "__main__":
    train()