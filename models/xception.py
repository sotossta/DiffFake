from torch import nn
import timm

class Xception(nn.Module):

    def __init__(self, extract_embeddings=False, multi_class=2):
        super(Xception, self).__init__()
        self.extract_embeddings = extract_embeddings
        
        # Load Xception from timm with pretrained weights
        self.net = timm.create_model('xception', pretrained=True)
        
        # Modify the final fully connected layer to match the number of classes
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, multi_class)

    def forward(self, x):
        if self.extract_embeddings:
            # Extract features up to the layer before the final fully connected layer
            x = self.net.forward_features(x)
            # Global Average Pooling
            x = nn.AdaptiveAvgPool2d((1, 1))(x)
            x = x.view(x.size(0), -1)
            return x
        else:
            x = self.net(x)
            return x
