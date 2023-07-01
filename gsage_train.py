import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from torch_geometric.utils import degree
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Cora dataset Citeseer, Pubmed
dataset = Planetoid(root="data/Planetoid/", name="Cora", transform=T.NormalizeFeatures())
dataset = dataset.shuffle()

# Access the graph data
data = dataset[0].to(device)

"""
# Remove degree to see the outcome 
# Get the node degrees
degrees = degree(data.edge_index[1], data.num_nodes)

# Convert degrees to one-hot encodings
degrees_onehot = torch.zeros((data.num_nodes, int(max(degrees)) + 1))
degrees_onehot.to(device).scatter_(1, degrees.unsqueeze(1).long(), 1)

# Expand the node features by concatenating the degrees
data.x = torch.cat([data.x.to(device), degrees_onehot.to(device)], dim=1)

# Normalize the node features
#data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0) 
"""

# Split dataset into train, validation, and test sets
train_idx, test_idx = train_test_split(range(data.num_nodes), test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

# Print the sizes of each set
print("Training set size:", len(train_idx))
print("Validation set size:", len(val_idx))
print("Test set size:", len(test_idx))

# Define GraphSAGELaplacian model
class GraphSAGEWithReg(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, dropout):
        super(GraphSAGEWithReg, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_features, hidden_features))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_features, hidden_features))
        self.layers.append(SAGEConv(hidden_features, out_features))

        self.laplacian_weight = nn.Parameter(torch.Tensor(hidden_features, hidden_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.laplacian_weight)

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)

        # Graph Laplacian regularization
        laplacian = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), size=(x.size(0), x.size(0)), device=device)
        laplacian = laplacian.to_dense()
        reg_loss = torch.trace(torch.matmul(laplacian, laplacian))

        return F.log_softmax(x, dim=1), reg_loss


# Define the warm-up function
def warmup_scheduler(optimizer, warmup_steps, init_lr):
    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return float(epoch) / float(warmup_steps)
        return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

# Set the number of warm-up steps and the initial learning rate
warmup_steps = 0 # 1000
init_lr = 0.01 # 0.0001 0.009300000000000001

num_epochs = 200
num_layers = 2

#in_features = dataset.num_features + degrees_onehot.size(1)
in_features = dataset.num_features
hidden_features = 16
out_features = dataset.num_classes
dropout=0

# Create model instance and move to CUDA device
model, data = GraphSAGEWithReg(in_features, hidden_features, out_features, num_layers, dropout).to(device), data.to(device)

# Create the optimizer with initial lr
#optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Create the warm-up scheduler
scheduler = warmup_scheduler(optimizer, warmup_steps, init_lr)


alpha = 0.01  # Scaling factor 
beta = 0.001  # Regularization coefficient 


def accuracy(output, labels):
    _, preds = output.max(dim=1)
    correct = preds.eq(labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
model.train()
for epoch in range(num_epochs):
    
    # Perform forward pass and calculate loss
    optimizer.zero_grad()
    out, reg_loss = model(data.x, data.edge_index)
    loss = F.nll_loss(F.log_softmax(out[train_idx], dim=1), data.y[train_idx])
    
    #loss += reg_loss  # Add regularization term to the loss
    loss += alpha*beta*reg_loss  # Add regularization term to the loss with scalar and Regularization coefficient 
    
    # Update model parameters
    loss.backward()
    optimizer.step()

    # Update learning rate using the scheduler
    scheduler.step()

 	
 	# Evaluate on training and validation set
    best_val_acc = 0
    with torch.no_grad():

    	model.eval()

    	out, _ = model(data.x, data.edge_index)
    	train_loss = F.nll_loss(F.log_softmax(out[train_idx], dim=1), data.y[train_idx])
    	train_acc = accuracy(out[train_idx], data.y[train_idx])

    	val_output, _ = model(data.x, data.edge_index)
    	val_loss =  F.nll_loss(F.log_softmax(out[val_idx], dim=1), data.y[val_idx])
    	val_acc = accuracy(val_output[val_idx], data.y[val_idx])

    	if val_acc > best_val_acc:
    		best_val_acc = val_acc
    		test_output, _ = model(data.x, data.edge_index)
    		test_acc = accuracy(test_output[test_idx], data.y[test_idx])

    	# Print current learning rate
    	#print(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {scheduler.get_lr()[0]}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    	print(f"Epoch [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")


    # Store the loss and accuracy values
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
   
    train_accuracies.append(float(train_acc))
    val_accuracies.append(float(val_acc))

# To get the last learning rate computed by the scheduler
# print(f'Last Learning Rate: {scheduler.get_last_lr()}')

print(f'Test Accuracy: {test_acc:.4f}')