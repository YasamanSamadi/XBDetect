# What the Code Does:
#Loads your RoninBridge.csv dataset.
#Creates a directed graph using the from and to columns as edges.
#Extracts a k-hop subgraph around any target node (address).
#Visualizes the subgraph using Matplotlib.
#Detects motifs like:
#Fan-In: Nodes with multiple incoming edges.
#Fan-Out: Nodes with multiple outgoing edges.
#Chain-Hopping: Paths that continue through multiple nodes.
#

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("RoninBridge.csv")

# Step 2: Print and inspect the column names
print("Columns in the dataset:", df.columns)

# Clean the column names to avoid errors (strip spaces and make lowercase)
df.columns = df.columns.str.strip().str.lower()

# Step 3: Adjust column names based on actual dataset
# Let's use 'source' and 'target' as the column names
if "source" not in df.columns:
    raise KeyError("The 'source' column is missing in the dataset. Please check the column names.")
if "target" not in df.columns:
    raise KeyError("The 'target' column is missing in the dataset. Please check the column names.")
if "contractaddress" not in df.columns:
    raise KeyError("The 'contractaddress' column is missing in the dataset. Please check the column names.")

# Step 4: Create a directed graph from the dataset
G = nx.from_pandas_edgelist(df, source="source", target="target", create_using=nx.DiGraph())

# Step 5: Define the function to extract k-hop subgraph
def extract_khop_subgraph(G, node, k=2):
    if node not in G:
        raise ValueError(f"Node {node} is not present in the graph.")
    
    subgraph_nodes = list(nx.single_source_shortest_path_length(G, node, cutoff=k).keys())
    return G.subgraph(subgraph_nodes)

# Step 6: Example usage - select a node
# Choose a specific node address from the dataset based on from, to, or contractAddress columns.
# Example: Select the first 'from' address or contract address

node_address = df['source'].iloc[0]  # Select the first sender address (from)
# Or if you want to use the contract address:
# node_address = df['contractaddress'].iloc[0]  # Select the first contract address

# Step 7: Check if the node exists in the dataset
if node_address in df['source'].values or node_address in df['target'].values or node_address in df['contractaddress'].values:
    if node_address in G:
        subgraph = extract_khop_subgraph(G, node_address, k=2)

        # Step 8: Print motif subgraphs and visualize
        if len(subgraph) > 0:
            # Print out details of the motifs in the subgraph
            print(f"Motif subgraphs around Node: {node_address}")
            
            # 1. Fan-In: Nodes with multiple incoming edges
            fan_in_nodes = [node for node in subgraph.nodes if len(list(subgraph.predecessors(node))) > 1]
            if fan_in_nodes:
                print(f"Fan-In Nodes: {fan_in_nodes}")
            
            # 2. Fan-Out: Nodes with multiple outgoing edges
            fan_out_nodes = [node for node in subgraph.nodes if len(list(subgraph.successors(node))) > 1]
            if fan_out_nodes:
                print(f"Fan-Out Nodes: {fan_out_nodes}")
            
            # 3. Chain-Hopping: Nodes involved in chain-hopping
            chain_hopping_paths = []
            for node in subgraph.nodes:
                successors = list(subgraph.successors(node))
                for successor in successors:
                    if len(list(subgraph.successors(successor))) > 0:
                        chain_hopping_paths.append((node, successor))
            if chain_hopping_paths:
                print(f"Chain-Hopping Paths: {chain_hopping_paths}")
            
            # Step 9: Visualize the extracted subgraph
            plt.figure(figsize=(12, 8))
            nx.draw(subgraph, with_labels=True, node_size=500, node_color="skyblue", font_size=10, edge_color="gray")
            plt.title(f"2-Hop Subgraph around Node: {node_address}")
            plt.show()
        else:
            print(f"No subgraph found for node: {node_address}")
    else:
        print(f"Node {node_address} not found in the graph.")
else:
    print(f"Node {node_address} does not exist in the dataset.")







