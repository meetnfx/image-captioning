import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
    def __len__(self):
        return len(self.itos)
    @staticmethod
    def tokenizer_eng(text):
        return str(text).lower().split()
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]
class COCODataset(Dataset):
    def __init__(self, root_dir, ann_file, split='train', transform=None, vocab=None, freq_threshold=5):
        """
        root_dir: The base image folder (e.g., 'data/images')
        ann_file: Path to Karpathy's dataset_coco.json
        split: 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        with open(ann_file, 'r') as f: # load karpathy
            self.coco_data = json.load(f)
        self.dataset_pairs = []
        all_captions = []
        for img in self.coco_data['images']: # get image image caption pair of a split
            img_split = img['split']
            valid_split = (split == 'train' and img_split in ['train', 'restval']) or (split == img_split) # use restval for train data
            if valid_split:
                folder = img['filepath'] # right folder
                filename = img['filename']
                full_path = os.path.join(self.root_dir, folder, filename)
                for sentence in img['sentences']: # 5 captions
                    caption = sentence['raw']
                    self.dataset_pairs.append((full_path, caption))
                    all_captions.append(caption)
        self.vocab = vocab # create vocab
        if self.vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(all_captions)
    def __len__(self):
        return len(self.dataset_pairs)


    def __getitem__(self, index):
        img_path, caption = self.dataset_pairs[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])
        return image, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets


def get_loader(root_dir, ann_file, split='train', transform=None, batch_size=32, num_workers=4, shuffle=True, pin_memory=True, vocab=None):
    dataset = COCODataset(root_dir, ann_file, split=split, transform=transform, vocab=vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    return loader, dataset
