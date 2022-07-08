from torch import nn


class LCNNModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, padding=1),
            #nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.fc1 = nn.Linear(in_features=48, out_features=64)
        self.drop = nn.Dropout2d(0.6)
        self.fc2 = nn.Linear(in_features=64, out_features=2)
    
    
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out
    
    
class LCNNModelMulti(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, padding=1),
            #nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.fc1 = nn.Linear(in_features=48, out_features=64)
        self.drop = nn.Dropout2d(0.6)
        self.fc2 = nn.Linear(in_features=64, out_features=13)
    
    
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out
    