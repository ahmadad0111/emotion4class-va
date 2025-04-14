
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        scores = self.W(x)  # (batch_size, seq_len, input_dim)
        scores = torch.tanh(scores)  # Apply non-linearity
        scores = self.v(scores).squeeze(-1)  # (batch_size, seq_len)
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)
        weighted_sum = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)
        return weighted_sum

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, projection_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_sizes[0], stride=stride, padding=(kernel_sizes[0] - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_sizes[1], stride=stride, padding=(kernel_sizes[1] - 1) // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)
        return out
    
class EEGNet1D(nn.Module):
    def __init__(self,in_channels, num_classes=4):
        super(EEGNet1D, self).__init__()

        # Initial Conv1D Layer
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=2, padding=0)  
        self.bn1 = nn.BatchNorm1d(32)
        
        # Residual Blocks with different kernel sizes
        self.res_block1 = ResidualBlock(in_channels=32, out_channels=64, kernel_sizes=[15, 15])
        self.res_block2 = ResidualBlock(in_channels=64, out_channels=128, kernel_sizes=[25, 25])
        self.res_block3 = ResidualBlock(in_channels=128, out_channels=128, kernel_sizes=[45, 45])

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Single Head Attention Layer
        self.attention = AttentionLayer(input_dim=128)
        # Projection head for contrastive learning
        self.proj_head = ProjectionHead(input_dim=128, projection_dim=128)
        # Fully Connected Layers
        # self.fc1 = nn.Linear(128, 1024)
        # #self.dropout1 = nn.Dropout(0.6)
        # self.fc2 = nn.Linear(1024, 512)
        # #self.dropout2 = nn.Dropout(0.6)
        # self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(128, num_classes)


    def forward(self, x):
        # print("shape of x")
        # print(x.shape)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.pool(x)
        x = x.transpose(1, 2)  # Transpose to (batch_size, seq_len, input_dim) for Attention Layer
        x = self.attention(x)  # Apply attention
        z = self.proj_head(x)

        # x = F.relu(self.fc1(x))
        # #x = self.dropout1(x)
        # x = F.relu(self.fc2(x))
        # #x = self.dropout2(x)
        # x = F.relu(self.fc3(x))
        x = self.fc4(z)
        #log softmax
        return F.log_softmax(x, dim=1),z
        #return torch.sigmoid(x)




class PredictorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PredictorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

