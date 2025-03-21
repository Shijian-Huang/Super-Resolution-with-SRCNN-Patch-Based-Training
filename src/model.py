import torch
import torch.nn as nn  # Import neural network module

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        """
        Initializes the SRCNN model.
        This model consists of three convolutional layers to enhance image resolution.
        """
        # First convolutional layer: Feature extraction
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        
        # Second convolutional layer: Non-linear mapping
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        
        # Third convolutional layer: Reconstruction (outputs a high-resolution image)
        self.layer3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)

        # ReLU activation function for non-linearity
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)

        return x