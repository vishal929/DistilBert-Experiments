# modules needed for knowledge graphs
# insipiration is from papers for GNN-LM and GreaseLM :
# 1. https://openreview.net/pdf?id=BS49l-B5Bql
# 2. https://arxiv.org/pdf/2201.08860.pdf

import torch

# firstly starting with greaseLM type architecture layer

# grease LM layers have the following submodules:
# 1. LM Layer -> standard language model transformer attention layer
# 2. GNN Layer -> Graph Neural Network layer (aggregation and message sending)
# 3. Mint Module -> Mint just stands for modality interaction, the authors just use a 2 layer MLP
# the authors code uses linear -> dropout -> layer norm -> gelu -> linear -> layer norm
class GreaseLMLayer(torch.nn.Module):
    def __init__(self, language_model_layer, gnn_layer, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.LM = language_model_layer
        self.GNN = gnn_layer

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # 2 layer mlp
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(in_dim,hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim,out_dim),
            torch.nn.LayerNorm(out_dim)
        )
    def forward(self,x):
        lm_res = self.LM(x)
        x = self.GNN(x)

        # multimodal fusion step with mlp
        x = torch.concat(lm_res,x)

        # running mlp then returning the new hidden state
        return self.MLP(x)