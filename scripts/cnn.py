import torch


class CNN(torch.nn.Module):
    
    def __init__(self, input_size=200):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.conv_out = input_size // 2 // 3
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=64,
                            kernel_size=3, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            #torch.nn.Dropout(.55),
            torch.nn.Conv1d(in_channels=64, out_channels=16,
                            kernel_size=5, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(3),
            #torch.nn.Dropout(.55),
            
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=16 * self.conv_out, out_features=50),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=50, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=4)
        )
        
    def forward(self, X):
        return self.layers(X)