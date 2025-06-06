#we initialize the heterogeneous graph with node types (addresses, bridges, chains) and edge types (transactions and bridges). We also define the attributes of nodes and edges.


import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset (replace with the path to your file)
df = pd.read_csv('RoninBridge.csv')

# Ensure the 'source' and 'target' columns are strings (if not, cast them)
df['source'] = df['source'].astype(str)
df['target'] = df['target'].astype(str)

# Create a mapping from source and target addresses to node indices
unique_addresses = pd.concat([df['source'], df['target']]).unique()
address_to_index = {address: idx for idx, address in enumerate(unique_addresses)}

# Create edge_index, mapping source and target addresses to node indices
edge_index = []
for _, row in df.iterrows():
    source_idx = address_to_index[row['source']]
    target_idx = address_to_index[row['target']]
    edge_index.append([source_idx, target_idx])

# Convert edge_index into a tensor
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Handle edge attributes (e.g., 'value' or other numeric features)
# For simplicity, we will just use 'value' here. Make sure to convert all columns to appropriate types.
# If the 'value' column is not numeric, convert it to float
df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Convert any non-numeric values to NaN

edge_attr = torch.tensor(df['value'].fillna(0).values, dtype=torch.float).view(-1, 1)  # Fill NaN values with 0

# Handle node features (if any). If you have other node features, use them here.
# For simplicity, we'll create random node features for each node.
#  replace this with actual node features from dataset.
node_features = torch.randn(len(unique_addresses), 10)  # 10-dimensional random features

# Create the PyTorch Geometric data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# Print the graph details (optional)
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of edges: {data.edge_index.shape[1]}')

# Print node features and edge attributes (optional)
print(f'Node features:\n{data.x}')
print(f'Edge indices:\n{data.edge_index}')
print(f'Edge attributes:\n{data.edge_attr}')

# Optionally, visualize the graph using NetworkX and Matplotlib
G_nx = nx.Graph()
for i in range(data.edge_index.shape[1]):
    u = data.edge_index[0, i].item()
    v = data.edge_index[1, i].item()
    G_nx.add_edge(u, v)

# Draw the graph
plt.figure(figsize=(12, 12))
nx.draw(G_nx, with_labels=True, node_size=50, font_size=10, node_color='skyblue', font_color='black')
plt.show()

# Optionally save the processed data (if needed)
# torch.save(data, 'processed_graph.pt')

# Optionally, save edge attributes to a file for further analysis
# df[['source', 'target', 'value']].to_csv('edge_attributes.csv', index=False)



