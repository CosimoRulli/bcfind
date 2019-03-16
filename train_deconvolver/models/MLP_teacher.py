import torch.nn as nn
import torch

class MLP_teacher(nn.Module):
    def __init__(self, in_out_size, hidden_size_1, hidden_size_2):
        super(MLP_teacher, self).__init__()
        self.fc1 = nn.Linear(in_out_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, in_out_size)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    #def __initialize_weights(self):
    #    pass

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(x)
        out = self.fc2(x)
        out = self.relu(x)
        out = self.fc3(x)
        out = self.sigm(x)
        return out


if __name__=="__main__":
    print("magi")