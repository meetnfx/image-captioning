import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the last two layers: AdaptiveAvgPool2d and Linear (fc)
        modules = list(resnet.children())[:-2] 
        self.resnet = nn.Sequential(*modules)

        # Projection layer to convert 2048 resnet features to embed_size
        self.projection = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        for name, param in self.resnet.named_parameters():
            if "layer4" in name and train_CNN:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, images):
        # resnet(images) -> [batch, 2048, 7, 7]
        features = self.resnet(images)

        # Reshape to [batch, 49, 2048]
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))

        # Project to [batch, 49, embed_size]
        features = self.projection(features)
        return features

    


