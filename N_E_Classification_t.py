import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("RoninBridge.csv")

# Step 2: Print and inspect the column names
print("Columns in the dataset:", df.columns)

# Clean the column names to avoid errors (strip spaces and make lowercase)
df.columns = df.columns.str.strip().str.lower()

# Step 3: Check if the 'source', 'target', or 'contractaddress' columns exist
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

# Step 6: Use the first 'source' address as the node (if available)
node_address = df['source'].iloc[0]  # Get the first source address from the dataframe

# Step 7: Verify that the node exists in the dataset (source, target, or contractaddress)
if node_address in df['source'].values or node_address in df['target'].values or node_address in df['contractaddress'].values:
    if node_address in G:
        subgraph = extract_khop_subgraph(G, node_address, k=2)

        # Step 8: Extract node features for classification (using simple degree as a feature here)
        node_features = np.array([G.degree(node) for node in subgraph.nodes])  # You can expand this with other features
        node_features = torch.tensor(node_features, dtype=torch.float32).view(-1, 1)  # Convert to tensor

        # Step 9: Classifier for node classification
        class Classifier(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(Classifier, self).__init__()
                self.fc = nn.Linear(in_channels, out_channels)
            
            def forward(self, x):
                return torch.sigmoid(self.fc(x))

        # Initialize the classifier
        classifier = Classifier(in_channels=1, out_channels=1)  # Assuming 1 feature per node for simplicity

        # Step 10: Train the model
        # For simplicity, we use random labels. Replace this with actual labels for laundering detection.
        labels = np.random.randint(0, 2, size=(len(node_features), 1))  # Random binary labels (0: non-laundering, 1: laundering)
        labels = torch.tensor(labels, dtype=torch.float32)

        # Step 11: Set up the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        # Step 12: Train the classifier (simple loop, 100 epochs)
        num_epochs = 100
        for epoch in range(num_epochs):
            # Forward pass
            outputs = classifier(node_features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Step 13: Make predictions on the node features
        with torch.no_grad():
            predictions = classifier(node_features)
            predicted_labels = predictions.round()  # Round to get binary labels (0 or 1)
        
        # Print predictions for the subgraph
        print(f"Predictions for the 2-hop subgraph around Node {node_address}:")
        for node, pred in zip(subgraph.nodes, predicted_labels):
            print(f"Node: {node}, Predicted Label (Laundering): {pred.item()}")

        # Step 14: Visualize the extracted subgraph
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        nx.draw(subgraph, with_labels=True, node_size=500, node_color="skyblue", font_size=10, edge_color="gray")
        plt.title(f"2-Hop Subgraph around Node: {node_address}")
        plt.show()

    else:
        print(f"Node {node_address} not found in the graph.")
else:
    print(f"Node {node_address} does not exist in the dataset.")
