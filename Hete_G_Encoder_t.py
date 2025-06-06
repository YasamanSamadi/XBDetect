#tests a graph neural network (GNN) model using Graph Convolutional Networks (GCNs) from PyTorch Geometric. use a simple graph convolution network (GCN) to encode the heterogeneous graph data.
# We'll assume that we have a predefined mapping for node and edge types, which we'll encode as distinct embeddings.
#The encoded node features are crucial for learning the multi-hop relationships and graph structure in subsequent layers.

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

# Load the dataset from "RoninBridge.csv"
df = pd.read_csv('RoninBridge.csv')

# Print the column names to ensure everything is as expected
print(f"Columns in the dataset: {df.columns}")

# Ensure the 'source' and 'target' columns are present
if 'source' not in df.columns or 'target' not in df.columns:
    raise ValueError("The dataset does not contain 'source' and 'target' columns.")

# Convert the 'source' and 'target' columns to strings
df['source'] = df['source'].astype(str)
df['target'] = df['target'].astype(str)

# Create a mapping from source and target addresses to node indices
unique_addresses = pd.concat([df['source'], df['target']]).unique()
address_to_index = {address: idx for idx, address in enumerate(unique_addresses)}

# Create edge_index (list of edges), mapping source and target addresses to node indices
edge_index = []
for _, row in df.iterrows():
    source_idx = address_to_index[row['source']]
    target_idx = address_to_index[row['target']]
    edge_index.append([source_idx, target_idx])

# Convert edge_index into a tensor
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Handle edge attributes (e.g., 'value')
df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Convert any non-numeric values to NaN
edge_attr = torch.tensor(df['value'].fillna(0).values, dtype=torch.float).view(-1, 1)  # Fill NaN values with 0

# Generate random node features (replace this with actual node features if available)
node_features = torch.randn(len(unique_addresses), 10)  # Example: 10-dimensional random features for each node

# Create the PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# Define input and output feature dimensions
in_channels = data.x.shape[1]  # Number of features per node (in our case, 10)
out_channels = 64  # Example size for embeddings

# Define the HeterogeneousGraphEncoder class
class HeterogeneousGraphEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeterogeneousGraphEncoder, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        return self.linear(x)

# Initialize the encoder
encoder = HeterogeneousGraphEncoder(in_channels, out_channels)

# Get the node embeddings
node_embeddings = encoder(data.x)

# Print the embeddings
print(f'Node embeddings shape: {node_embeddings.shape}')
print(node_embeddings)


