import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import pandas as pd

# Load the dataset
df = pd.read_csv('RoninBridge.csv')

# Convert string node IDs to numeric values
df['source'], source_map = pd.factorize(df['source'])
df['target'], target_map = pd.factorize(df['target'])

# Prepare the edge index (source, target) from the DataFrame
edge_index = torch.tensor(df[['source', 'target']].values.T, dtype=torch.long)

# Dummy node features
num_nodes = len(source_map) + len(target_map)
x = torch.randn((num_nodes, 16))

# Create the PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index)

# VGAE Encoder
class VGAEEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv_mu = GCNConv(out_channels, out_channels)
        self.conv_logstd = GCNConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd

# Instantiate the VGAE Model
in_channels = data.num_features
out_channels = 64

encoder = VGAEEncoder(in_channels, out_channels)
vgae_model = VGAE(encoder)
optimizer = torch.optim.Adam(vgae_model.parameters(), lr=0.01)

# Loss Function
def vgae_loss(z, pos_edge_index, neg_edge_index=None):
    # Positive reconstruction loss
    pos_loss = -torch.log(
        torch.sigmoid((z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)) + 1e-15
    ).mean()

    # Negative reconstruction loss
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(
            pos_edge_index, num_nodes=z.size(0), num_neg_samples=pos_edge_index.size(1)
        )
    neg_loss = -torch.log(
        1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)) + 1e-15
    ).mean()

    # Total reconstruction loss
    recon_loss = pos_loss + neg_loss

    # KL Divergence loss
    kl_loss = -0.5 * torch.mean(1 + vgae_model.__logstd__ - vgae_model.__mu__**2 - vgae_model.__logstd__.exp())

    return recon_loss + kl_loss

# Training Loop
vgae_model.train()
for epoch in range(100):
    optimizer.zero_grad()
    z = vgae_model.encode(data.x, data.edge_index)
    loss = vgae_loss(z, data.edge_index)
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
    loss.backward()
    optimizer.step()
