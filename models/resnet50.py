from torch import nn
from torchvision import models

class ResNet50(nn.Module):

    def __init__(self, extract_embeddings=False, multi_class=2):
        super(ResNet50, self).__init__()
        self.extract_embeddings = extract_embeddings
        
        # Load ResNet-50 with pretrained weights
        self.net = models.resnet50(pretrained=True)
        
        # Modify the final fully connected layer to match the number of classes
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, multi_class)
        
    def forward(self, x):
        if self.extract_embeddings:
            # Extract features up to the layer before the final fully connected layer
            x = nn.Sequential(*list(self.net.children())[:-1])(x)
            # Flatten the output of the last convolutional block
            x = x.view(x.size(0), -1)
            return x
        else:
            x = self.net(x)
            return x
