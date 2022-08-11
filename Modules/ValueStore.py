# defining modules needed for a value store
# inspiration is from paper Memorizing Transformers: https://arxiv.org/pdf/2203.08913.pdf
# we will use faiss approximate k-nn

import torch
import faiss

class GatedLocalValueAttention(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # scalar gating parameter called gamma
        self.gamma = torch.tensor(in_dim)

    def forward(self, localAttention, keyValueStoreAttention):
        # simply using the parameter to scale local attention and keyValueStoreAttention
        # this is just "choosing" between using the transformer attention vs key value store knn attention
        # via element wise multiplication
        gate = torch.sigmoid(self.gamma)
        return localAttention * gate + keyValueStoreAttention * (1-gate)


# purpose of this method is to perform both local attention and memory knn attention
# then use the gating module to weight both memory and local representations
# intended to be the last major layer of a transformer
class KeyValueStoreLayer(torch.nn.Module):
    """
        memory_size: number of key,value token pairs possible
        top_k: number of candidates to retrieve in k-nn algorithm
    """
    def __init__(self, memory_size, top_k, in_dim, out_dim):
        super().__init__()
        # initializing the key value store
        self.memory =
        self.localAttention = torch.nn.MultiheadAttention()
        self.memoryAttention = torch.nn.MultiheadAttention()
        self.gate = GatedLocalValueAttention()

    def forward(self, query):
        # computing local attention (self attention in this case)
        locAttention = self.localAttention.forward(query,query,query)
        # using knn on key value store, computing attention, and then updating the store
        memKeys = None
        memVals = None
        memAttention = self.memoryAttention.forward()

        finalVal = self.gate(locAttention, memAttention)

        # updating the key value store to include the new key value pairs

        return finalVal

    def clearStore(self):
        # clearing memory (i.e when training on a new document)
        pass