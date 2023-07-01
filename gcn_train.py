import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, PPI, WikiCS, Flickr, Yelp
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define GCN model with graph Laplacian regularization
class GCNWithRegularization(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, dropout):
        super(GCNWithRegularization, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_features, hidden_features))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_features, hidden_features))
        self.layers.append(GCNConv(hidden_features, out_features))

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


# Load the Cora dataset Citeseer, Pubmed
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Planetoid')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
dataset = dataset.shuffle()
data = dataset[0].to(device)

'''
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
'''

# Split dataset into train, validation, and test sets
train_mask, test_mask = train_test_split(range(data.num_nodes), test_size=0.2, random_state=42)
train_mask, val_mask = train_test_split(train_mask, test_size=0.2, random_state=42)


# Print the sizes of each set
print("Training set size:", len(train_mask))
print("Validation set size:", len(val_mask))
print("Test set size:", len(test_mask))



# Create the Graph Laplacian encoder model
#remove + degree anytime to remove degree at the top
in_features = dataset.num_features
#in_features = dataset.num_features + degrees_onehot.size(1)
hidden_features = 16
out_features = dataset.num_classes
num_layers = 2
dropout=0

# Set the number of warm-up steps and the initial learning rate
num_epochs = 300
warmup_steps = 0 # 100
init_lr = 0.01 # 0.01

model, data = GCNWithRegularization(in_features, hidden_features, out_features, num_layers, dropout).to(device), data.to(device)


# Define the optimizer and loss function
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

# Lists to store training and validation loss/accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output, reg_loss = model(data.x, data.edge_index)
    loss = F.nll_loss(F.log_softmax(output[train_mask], dim=1), data.y[train_mask])
    
    # Add the regularization term to the loss
    loss += alpha*beta*reg_loss
    
    loss.backward()
    optimizer.step()

    # Update learning rate using the scheduler
    scheduler.step()

    # Evaluate on training and validation set
    best_val_acc = 0
    model.eval()
    with torch.no_grad():
        train_output, _ = model(data.x, data.edge_index)
        train_loss = F.nll_loss(F.log_softmax(train_output[train_mask], dim=1), data.y[train_mask])
        train_acc = accuracy(train_output[train_mask], data.y[train_mask])
        
        val_output = model(data.x, data.edge_index)
        val_loss =  F.nll_loss(F.log_softmax(output[val_mask], dim=1), data.y[val_mask])
        val_acc = accuracy(output[val_mask], data.y[val_mask])

        
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_output, _ = model(data.x, data.edge_index)
            test_acc = accuracy(test_output[test_mask], data.y[test_mask])

        print(f'Epoch: [{epoch+1}/{num_epochs}], Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    
    # Store the loss and accuracy values
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
   
    train_accuracies.append(float(train_acc))
    val_accuracies.append(float(val_acc))

print(f'Test Accuracy: {test_acc:.4f}')
print(' ')


# Plotting the training and validation curves
plt.figure(figsize=(10, 5))
epochs = range(num_epochs)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


"""
# Obtain the node embeddings
model.eval()
with torch.no_grad():
    node_embeddings = model(data.x, data.edge_index).cpu()

# Perform t-SNE embedding
tsne = TSNE(n_components=2)
node_embeddings_tsne = tsne.fit_transform(node_embeddings.numpy())

# Plot the t-SNE visualization
plt.scatter(node_embeddings_tsne[:, 0], node_embeddings_tsne[:, 1], c=data.y.cpu().numpy(), 
    cmap='viridis', s=10)
#plt.xlabel('t-SNE Dimension 1')
#plt.ylabel('t-SNE Dimension 2')
# plt.title('Visualization of Node Embeddings' )
# plt.colorbar()
plt.show()


# # Calculate information-to-noise ratio for the validation set
# def calculate_information_noise_ratio(embeddings, labels):
#     distances = pairwise_distances(embeddings, metric='correlation')
#     label_distances = distances[labels == 1]
#     noise_distances = distances[labels == 0]
#     avg_label_distance = label_distances.mean()
#     avg_noise_distance = noise_distances.mean()
#     information_noise_ratio = avg_noise_distance / avg_label_distance
#     #information_noise_ratio = (1 - information_noise_ratio_)
#     return information_noise_ratio

# # Calculate information-to-noise ratio for the validation set
# information_noise_ratio = calculate_information_noise_ratio(node_embeddings_tsne, data.y.cpu())

# # Print the information-to-noise ratio
# print(f"Information-to-Noise Ratio: {information_noise_ratio:.4f}")


# from information_noise_ratio import _calculate_information_noise_ratio
# # Usage example
# info_noise_ratio = _calculate_information_noise_ratio(node_embeddings_tsne, data.y.cpu())
# print(f"Information-to-Noise Ratio: {info_noise_ratio:.4f}")
"""