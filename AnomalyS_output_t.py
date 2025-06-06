import pandas as pd
import torch
import networkx as nx
import numpy as np
import torch.nn as nn

# Load the dataset
df = pd.read_csv("RoninBridge.csv")

# Ensure 'value' column is numeric (convert any non-numeric strings to NaN)
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Example function to extract a k-hop subgraph around a node
def extract_khop_subgraph(G, node, k):
    subgraph_nodes = list(nx.single_source_shortest_path_length(G, node, cutoff=k).keys())
    subgraph = G.subgraph(subgraph_nodes).copy()
    return subgraph

# Create graph (networkx)
G = nx.from_pandas_edgelist(df, source='source', target='target')

# Select node address based on the 'source' column (for example, using the first 'source' address from the dataframe)
first_sender = df['source'].iloc[0]

# Extract the k-hop subgraph
k = 2  # Set the hop distance
subgraph = extract_khop_subgraph(G, first_sender, k)

# Dummy node embeddings (e.g., from HAN layers or any embedding method)
# Here, I'm assuming the embeddings are random for illustration purposes.
# Replace this with your actual node embeddings.
node_embeddings_han = torch.rand(len(subgraph.nodes), 128)  # Assuming 128-dimensional embeddings

# Step 7: Anomaly Scoring based on degree, transaction patterns, and bridge volume
def compute_anomaly_scores(data, node_embeddings, subgraph):
    # Example scoring: degree, transaction irregularity, and bridge volume
    
    # Degree-based score (number of neighbors in the subgraph)
    degree = torch.tensor([G.degree(node) for node in subgraph.nodes])

    # Transaction pattern score (sum of transaction values for each node)
    transaction_pattern = torch.tensor([
        data[data['source'] == node]['value'].sum() + data[data['target'] == node]['value'].sum() 
        for node in subgraph.nodes
    ])

    # Bridge volume score (sum of transaction values associated with the contract address for each node)
    bridge_volume = torch.tensor([
        data[data['contractAddress'] == node]['value'].sum() for node in subgraph.nodes
    ])

    # Combine the scores into a final anomaly score
    anomaly_score = degree + transaction_pattern + bridge_volume
    return anomaly_score

# Compute anomaly scores
anomaly_scores = compute_anomaly_scores(df, node_embeddings_han, subgraph)

# Step 8: Binary Classification Output (laundering or not)
# Using a threshold of 0.5 to classify as laundering (1) or not laundering (0)
binary_output = (anomaly_scores > 0.5).float()

# Print results for demonstration
print("Anomaly Scores:", anomaly_scores)
print("Binary Classification Output (0: Not laundering, 1: Laundering):", binary_output)
