import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATv2Conv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("RoninBridge.csv")

# Feature Engineering with reduced multicollinearity (drop timeStamp and blockNumber)
def feature_engineering(df):
    transaction_volume = df.groupby('source')['value'].sum().to_dict()
    gas_price = df.groupby('source')['gasPrice'].mean().to_dict()
    time_diff = (
        df.sort_values(by=['source', 'timeStamp'])
        .groupby('source')['timeStamp']
        .apply(lambda x: x.diff().mean() if len(x) > 1 else 0)
        .to_dict()
    )
    return {'transaction_volume': transaction_volume, 'gas_price': gas_price, 'time_diff': time_diff}

features_dict = feature_engineering(df)

# Preprocess graph data (drop blockNumber and timeStamp)
def preprocess_graph(df, features_dict):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row["source"], type="address")
        G.add_node(row["target"], type="address")
        G.add_edge(row["source"], row["target"], value=row["value"])
    for node in G.nodes:
        G.nodes[node]['transaction_volume'] = features_dict['transaction_volume'].get(node, 0)
        G.nodes[node]['gas_price'] = features_dict['gas_price'].get(node, 0)
        G.nodes[node]['time_diff'] = features_dict['time_diff'].get(node, 0)
    return G

G = preprocess_graph(df, features_dict)

# Generate node features
def generate_features(G):
    features = []
    for node in G.nodes:
        degree = float(G.degree[node])
        total_value = sum(
            float(data.get("value", 0)) for _, _, data in G.edges(node, data=True)
        )
        features.append([degree, total_value])
    return torch.tensor(features, dtype=torch.float)

node_feats = generate_features(G)

# Standardize node features
scaler = StandardScaler()
node_feats = torch.tensor(scaler.fit_transform(node_feats), dtype=torch.float)

# Create edge index
node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}
edge_index = torch.tensor(
    [[node_to_idx[u], node_to_idx[v]] for u, v in G.edges], dtype=torch.long
).t().contiguous()

# Generate labels (binary classification task)
labels = torch.randint(0, 2, (len(G.nodes),))

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
node_feats_np = node_feats.numpy()
labels_np = labels.numpy()
node_feats_np, labels_np = smote.fit_resample(node_feats_np, labels_np)
node_feats = torch.tensor(node_feats_np, dtype=torch.float)
labels = torch.tensor(labels_np, dtype=torch.long)

# Split dataset
def split_dataset(num_nodes, labels):
    idx = np.arange(num_nodes)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        idx, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
    )
    return train_idx, val_idx, test_idx

train_idx, val_idx, test_idx = split_dataset(len(node_feats), labels)
train_mask = torch.tensor(train_idx, dtype=torch.long)
val_mask = torch.tensor(val_idx, dtype=torch.long)
test_mask = torch.tensor(test_idx, dtype=torch.long)

# Data object
data = Data(x=node_feats, edge_index=edge_index)

# Modified GNN model with more layers, dropout, and layer normalization
class EnhancedGNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EnhancedGNNModel, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.gat = GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels * 2)  # Additional fully connected layer
        self.fc2 = nn.Linear(hidden_channels * 2, out_channels)     # Output layer
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden_channels)  # Added layer normalization

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.layer_norm(x)  # Apply layer normalization
        x = F.relu(self.gcn2(x, edge_index))
        x = self.layer_norm(x)
        x = self.gat(x, edge_index)
        x = F.relu(self.fc1(x))  # Apply the fully connected layer
        x = self.dropout(x)
        return self.fc2(x)

# Initialize model, loss function, and optimizer
model = EnhancedGNNModel(in_channels=node_feats.size(1), hidden_channels=256, out_channels=2)
loss_weights = torch.tensor([(labels == 0).sum() / len(labels), (labels == 1).sum() / len(labels)], dtype=torch.float)
loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

# Training and evaluation functions
def train(model, data, train_mask, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        y_true = labels[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        
        # Precision-Recall AUC
        precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall_pr, precision_pr)
        
        return accuracy, precision, recall, f1, roc_auc, pr_auc

# Training loop with adjusted epochs
num_epochs = 200  # Increased epochs for better performance

# Initialize variables to store cumulative metrics
total_accuracy = 0
total_precision = 0
total_recall = 0
total_f1 = 0
total_roc_auc = 0
total_pr_auc = 0

# Training loop
for epoch in range(num_epochs):
    loss = train(model, data, train_mask, optimizer, loss_fn)
    scheduler.step()  # Update the learning rate
    if epoch % 10 == 0:  # Adjust frequency if necessary
        accuracy, precision, recall, f1, roc_auc, pr_auc = evaluate(model, data, test_mask)
        
        # Accumulate metrics for averaging later
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_roc_auc += roc_auc
        total_pr_auc += pr_auc

# Calculate average metrics
average_accuracy = total_accuracy / (num_epochs // 10)
average_precision = total_precision / (num_epochs // 10)
average_recall = total_recall / (num_epochs // 10)
average_f1 = total_f1 / (num_epochs // 10)
average_roc_auc = total_roc_auc / (num_epochs // 10)
average_pr_auc = total_pr_auc / (num_epochs // 10)

print(f"Average Accuracy: {average_accuracy:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average F1 Score: {average_f1:.4f}")
print(f"Average ROC-AUC: {average_roc_auc:.4f}")
print(f"Average Precision-Recall AUC: {average_pr_auc:.4f}")

# Linear Regression Boosting
def linear_regression_boosting(data, labels, train_idx, test_idx):
    model.eval()
    with torch.no_grad():
        gnn_out = model(data.x, data.edge_index).cpu().numpy()
    lr = LinearRegression()
    lr.fit(gnn_out[train_idx], labels[train_idx].cpu().numpy())
    lr_pred = lr.predict(gnn_out[test_idx])
    lr_pred = (lr_pred > 0.5).astype(int)
    accuracy = accuracy_score(labels[test_idx].cpu().numpy(), lr_pred)
    print(f"Linear Regression Boosting Accuracy: {accuracy:.4f}")

linear_regression_boosting(data, labels, train_idx, test_idx)
