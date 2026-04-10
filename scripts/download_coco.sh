#!/bin/bash

# go to data
cd ../data

echo "Downloading Karpathy Splits JSON"
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
mv dataset_coco.json annotations/
rm caption_datasets.zip

echo "Downloading MS COCO 2014 Training Images (13GB)..."
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d images/
rm train2014.zip

echo "Downloading MS COCO 2014 Validation Images (6GB)..."
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d images/
rm val2014.zip

echo "Download and extraction done"