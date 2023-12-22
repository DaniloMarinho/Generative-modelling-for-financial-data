import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import H

class Generator(nn.Module):
    def __init__(self, latent_dim, g_hidden_dim, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(latent_dim, g_hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

        # self.fc5 = nn.Linear(2*g_output_dim, g_output_dim)

    # forward method
    def forward(self, x):
        # z = x
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return self.fc4(x)

    # # forward method - pure evgan
    # def forward(self, x):
    #     z = x
    #     x = F.leaky_relu(self.fc1(x), 0.2)
    #     x = F.leaky_relu(self.fc2(x), 0.2)
    #     x = F.leaky_relu(self.fc3(x), 0.2)
    #     # enforce positivity on x for H
    #     x = F.relu(self.fc4(x))
    #     return torch.stack([H(z[:, i], x[:, i]) for i in range(x.shape[1])]).t()
        
    # forward method - mixed
    # def forward(self, x):
    #     z = x
    #     x = F.leaky_relu(self.fc1(x), 0.2)
    #     x = F.leaky_relu(self.fc2(x), 0.2)
    #     x = F.leaky_relu(self.fc3(x), 0.2)
    #     # enforce positivity on x for H, and dimensionality up to z
    #     x = self.fc4(x)
    #     x = torch.concat([torch.stack([H(z[:, i], F.relu(x)[:, i]) for i in range(x.shape[1])]).t(), x], axis=1)
    #     return self.fc5(x)

class Discriminator(nn.Module):
    def __init__(self, d_input_dim, d_hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, d_hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))
