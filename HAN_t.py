# message pardding layer

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as pyg_nn
import pandas as pd

# Load the dataset
df = pd.read_csv('RoninBridge.csv')

# Ensure 'value' and 'gas' are numeric, forcing non-numeric values to NaN
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df['gas'] = pd.to_numeric(df['gas'], errors='coerce')

# Handle NaN values by replacing them with 0 or another strategy (e.g., median or mean)
df['value'].fillna(0, inplace=True)
df['gas'].fillna(0, inplace=True)

# Extracting the source and target nodes from the dataset
df['source'] = df['source'].astype(str)
df['target'] = df['target'].astype(str)

# Create a mapping of unique node identifiers (string to integer)
node_mapping = {node: idx for idx, node in enumerate(pd.concat([df['source'], df['target']]).unique())}

# Convert source and target nodes to numeric values using the node mapping
df['source'] = df['source'].map(node_mapping)
df['target'] = df['target'].map(node_mapping)

# Creating the edge_index (pairing the source and target addresses as edges)
edge_index = torch.tensor(df[['source', 'target']].values.T, dtype=torch.long)

# Now, ensure that the 'value' and 'gas' columns are numeric and create node_feats
node_feats = torch.tensor(df[['value', 'gas']].values, dtype=torch.float)

# If you have no edge attributes, use a placeholder
edge_attr = torch.tensor([1] * edge_index.size(1), dtype=torch.float).view(-1, 1)  # Example: dummy edge attributes

# GATLayer (Graph Attention Layer) class
class GATLayer(pyg_nn.GATConv):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GATLayer, self).__init__(in_channels, out_channels, heads=heads, concat=True)

    def forward(self, x, edge_index, edge_attr=None):
        return super(GATLayer, self).forward(x, edge_index)

# HANModel class with GATConv
class HANModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HANModel, self).__init__()
        self.layer1 = GATLayer(in_channels, out_channels, heads=4)  # First layer with 4 heads
        self.layer2 = GATLayer(out_channels * 4, out_channels, heads=4)  # Second layer, with multiplied output size (since concat=True)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Pass data through the layers
        x = self.layer1(x, edge_index, edge_attr)
        x = self.layer2(x, edge_index, edge_attr)
        return x

# Prepare data for the HAN model
class HANData(torch_geometric.data.Data):
    def __init__(self, x, edge_index, edge_attr):
        super(HANData, self).__init__()
        self.x = x  # Node features
        self.edge_index = edge_index  # Edge index
        self.edge_attr = edge_attr  # Edge attributes (if any)

# Create a HANData object with the required attributes
data = HANData(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)

# Initialize the HAN model with 2 input features (value, gas) and an output feature size of 128
han_model = HANModel(in_channels=2, out_channels=128)  # Corrected in_channels to 2
node_embeddings_han = han_model(data)  # Get node embeddings after HAN layers

# Optionally print node embeddings
print(node_embeddings_han)



