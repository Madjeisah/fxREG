import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import degree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define GCN model with graph Laplacian regularization
class GCNWithReg(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, dropout):
        super(GCNWithReg, self).__init__()

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


# Define GraphSAGE model with graph Laplacian regularization
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