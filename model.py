import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm

class Decoder(nn.Module):
    def __init__(
        self,
        args,
        dropout_prob=0.2,
    ):
        super(Decoder, self).__init__()

        # **** YOU SHOULD ADD TRAINING CODE HERE, CURRENTLY IT IS INCORRECT ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # Relu layers, Dropout layers and a tanh layer.
        self.fc1 = weight_norm(nn.Linear(3, 512))
        self.fc2 = weight_norm(nn.Linear(512, 512))
        self.fc3 = weight_norm(nn.Linear(512, 512))
        self.fc4 = weight_norm(nn.Linear(512, 509))
        self.fc5 = weight_norm(nn.Linear(512, 512))
        self.fc6 = weight_norm(nn.Linear(512, 512))
        self.fc7 = weight_norm(nn.Linear(512, 512))
        self.fc8 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        # self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.th = nn.Tanh()
        # ***********************************************************************

    # input: N x 3
    def forward(self, input):

        # **** YOU SHOULD ADD TRAINING CODE HERE, CURRENTLY IT IS INCORRECT ****
        # You should implement the network forward passing here
        x = self.fc1(input)
        x = self.relu(x)
        # dropout = nn.Dropout(p=self.dropout_prob)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        # dropout = nn.Dropout(p=self.dropout_prob)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        # dropout = nn.Dropout(p=self.dropout_prob)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        # dropout = nn.Dropout(p=self.dropout_prob)
        x = self.dropout(x)

        x = torch.cat((x, input), dim=1)
        x = self.fc5(x)
        x = self.relu(x)
        # dropout = nn.Dropout(p=self.dropout_prob)
        x = self.dropout(x)

        x = self.fc6(x)
        x = self.relu(x)
        # dropout = nn.Dropout(p=self.dropout_prob)
        x = self.dropout(x)

        x = self.fc7(x)
        x = self.relu(x)
        # dropout = nn.Dropout(p=self.dropout_prob)
        x = self.dropout(x)        
        
        x = self.fc8(x)
        x = self.th(x)
        # ***********************************************************************


        return x
