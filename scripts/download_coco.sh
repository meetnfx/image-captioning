#!/bin/bash

# go to data
DATA_DIR=/content/drive/MyDrive/image-captioning/data

mkdir -p "$DATA_DIR/annotations"
mkdir -p "$DATA_DIR/images"

cd "$DATA_DIR"

echo "Downloading Karpathy Splits JSON..."
# Using the standard Karpathy split link
wget -c http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip -o caption_datasets.zip
mv dataset_coco.json annotations/
rm caption_datasets.zip

echo "Downloading MS COCO 2014 Training Images (13GB)..."
wget -c http://images.cocodataset.org/zips/train2014.zip
unzip -q train2014.zip -d images/
rm train2014.zip

echo "Downloading MS COCO 2014 Validation Images (6GB)..."
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip -q val2014.zip -d images/
rm val2014.zip

echo "Download and extraction complete!"
