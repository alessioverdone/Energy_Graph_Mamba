# energy-graph-mamba


[![PyPI version](https://badge.fury.io/py/x-transformers.svg)](https://badge.fury.io/py/x-transformers)

An implementation of Mamba architecture on forecasting temporal graph time series on energy tasks.

## Install

```bash
$ pip install energy-graph-mamba
```

## Usage


PyTorch Geometric Temporal makes implementing Dynamic and Temporal Graph Neural Networks quite easy. For example, this is all it takes to implement a recurrent graph convolutional network with two consecutive graph convolutional GRU cells and a linear layer:

```python
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

class RecurrentGCN(torch.nn.Module):

    def __init__(self, node_features, num_classes):
        super(RecurrentGCN, self).__init__()
        self.recurrent_1 = GConvGRU(node_features, 32, 5)
        self.recurrent_2 = GConvGRU(32, 16, 5)
        self.linear = torch.nn.Linear(16, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
```


## Citations

```bibtex
@misc{seo2016structured,
      title={Structured Sequence Modeling with Graph Convolutional Recurrent Networks}, 
      author={Youngjoo Seo and Michaël Defferrard and Pierre Vandergheynst and Xavier Bresson},
      year={2016},
      eprint={1612.07659},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```



