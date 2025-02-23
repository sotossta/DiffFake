from torch import nn
from efficientnet_pytorch import EfficientNet

class Detector(nn.Module):

    def __init__(self,extract_embeddings=False,multi_class=2):
        super(Detector, self).__init__()
        self.extract_embeddings = extract_embeddings
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=multi_class)
        
    def forward(self, x):
        
        if self.extract_embeddings:
            # Extract features up to the layer before the final fully connected layer
            x = self.net.extract_features(x)
            # Global Pooling
            x = self.net._avg_pooling(x)
            return x
        else:
            x = self.net(x)
            return x