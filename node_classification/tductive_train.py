import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import trange
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import os.path as osp

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, PPI, WikiCS, Flickr, Yelp
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt


from tductive_model import *


print('')
print('Initiating...')
#time.sleep(60)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
	"""
	Utility function to set seed values for RNG for various modules
	"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False



class Options:

	def __init__(self):
		self.parser = argparse.ArgumentParser(description="Graph Classification")
		self.parser.add_argument("--epochs", dest="epochs", action="store", default=200, type=int)
		self.parser.add_argument("--lr", dest="lr", action="store", default=5e-3, type=float)
		self.parser.add_argument("--init_lr", dest="init_lr", action="store", default=1e-2, type=float)
		self.parser.add_argument("--warmup_steps", dest="warmup_steps", action="store", default=0, type=float)
		self.parser.add_argument("--feat_dim", dest="feat_dim", action="store", default=16, type=int)
		self.parser.add_argument("--layers", dest="layers", action="store", default=2, type=int)
		self.parser.add_argument("--alpha", dest="alpha", action="store", default=0.01, type=int)
		self.parser.add_argument("--beta", dest="beta", action="store", default=0.001, type=int)
		self.parser.add_argument("--dataset", dest="dataset", action="store", required=True, type=str,
			choices=["cora", "citeseer", "pubmed"])
		self.parser.add_argument("--model", dest="model", action="store", default="gcn", type=str,
			choices=["gcn", "graphsage"])
		

		self.parse()

	def parse(self):
		self.opts = self.parser.parse_args()

	def __str__(self):
		return ("All Options:\n" + "".join(["-"] * 45) + "\n" + "\n".join(["{:<18} -------> {}".format(k, v) for k, v in vars(self.opts).items()]) + "\n" + "".join(["-"] * 45) + "\n")


# Load the Cora dataset Citeseer, Pubmed
def load_dataset(name):
	if name == "cora":
		dataset = Planetoid(root="data/Planetoid/", name="Cora", transform=T.NormalizeFeatures())
	elif name == "citeseer":
		dataset = Planetoid(root="data/Planetoid/", name="Citeseer", transform=T.NormalizeFeatures())
	elif name == "pubmed":
		dataset = Planetoid(root="data/Planetoid/", name="Pubmed", transform=T.NormalizeFeatures())

	return dataset


# Define the warm-up function
def warmup_scheduler(optimizer, warmup_steps, init_lr):
    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return float(epoch) / float(warmup_steps)
        return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler





def main(args):
	dataset = load_dataset(args.dataset)
	dataset = dataset.shuffle()
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
	train_mask, test_mask = train_test_split(range(data.num_nodes), test_size=0.2, random_state=42)
	train_mask, val_mask = train_test_split(train_mask, test_size=0.2, random_state=42)

	# Print the sizes of each set
	print("Training set size:", len(train_mask))
	print("Validation set size:", len(val_mask))
	print("Test set size:", len(test_mask))

	#in_features = dataset.num_features + degrees_onehot.size(1)
	in_features = dataset.num_features
	out_features = dataset.num_classes
	dropout = 0

	if args.model == "gcn":
		model = GCNWithReg(in_features, hidden_features=args.feat_dim, 
			out_features = out_features, num_layers=args.layers, dropout=dropout).to(device)
	elif args.model == "graphsage":
		model = GraphSAGEWithReg(in_features, hidden_features=args.feat_dim, 
			out_features = out_features, num_layers=args.layers, dropout=dropout).to(device)
	else: 
		print('Enter the right model')

	optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

	# Create the warm-up scheduler
	scheduler = warmup_scheduler(optimizer, args.warmup_steps, args.init_lr)


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
	for epoch in range(args.epochs):
		optimizer.zero_grad()
		output, reg_loss = model(data.x, data.edge_index)
		loss = F.nll_loss(F.log_softmax(output[train_mask], dim=1), data.y[train_mask])

		# Add the regularization term to the loss
		loss += args.alpha*args.beta*reg_loss

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

			print(f'Epoch: [{epoch+1}/{args.epochs}], Learning Rate: {scheduler.get_lr()[0]}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

		# Store the loss and accuracy values
		train_losses.append(train_loss.item())
		val_losses.append(val_loss.item())

		train_accuracies.append(float(train_acc))
		val_accuracies.append(float(val_acc))

	print(f'Test Accuracy: {test_acc:.4f}')
	print(' ')   

if __name__ == "__main__":

	set_seed(0)
	args = Options()
	print(args)

	main(args.opts)
	
	