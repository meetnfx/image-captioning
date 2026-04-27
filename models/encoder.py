import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # use resnet50 already done
        modules = list(resnet.children())[:-2]  # remove final layer, so we get raw features
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size) # Lin layer map 2048 resnet feature to decoder size
        self.norm = nn.LayerNorm(embed_size)
        for name, param in self.resnet.named_parameters(): # freeze resnet weight maybe to save ram
            if "layer4" in name and train_CNN: # misc change final layer if needed
                param.requires_grad = True
            else:
                param.requires_grad = False
    def forward(self, images):
        features = self.resnet(images)   # IMPORTANT [batch_size, 3, 224, 224] input
        features = features.view(features.size(0), features.size(1), -1)  # flat feature batch size, 2048, 49
        features = features.permute(0, 2, 1)
        features = self.norm(self.linear(features))
        return features


    


