# fxREG
fxREG: Frobenious norm Graph Laplacian Regularization for Node and Graph Classification



## Downstream Tasks
For node classification, run:
`python tductive_train.py --dataset cora --model graphsage --layers 2 --epochs 200 --feat_dim 32`

For graph classification, run:
`python hraph_train.py --dataset nci1 --model gcn --layers 2 --epochs 200 --feat_dim 32`
