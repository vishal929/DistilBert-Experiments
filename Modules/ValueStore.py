# defining modules needed for a value store
# inspiration is from paper Memorizing Transformers: https://arxiv.org/pdf/2203.08913.pdf
# we will use faiss approximate k-nn

import torch
import faiss
import os
import numpy as np

# external file for indexing that needs to spill to disk
db_file_path = os.path.join(os.getcwd(),"Resources","value_store")

# this represents the gating module that "chooses" between model values and memory store values
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
        # initializing the key value store -> in_dim *2 for memory dimension since we are storing key,value pairs
        self.memory = MemoryStore(memory_size,top_k,in_dim*2)
        self.localAttention = torch.nn.MultiheadAttention()
        self.memoryAttention = torch.nn.MultiheadAttention()
        self.gate = GatedLocalValueAttention(out_dim)

    def forward(self, query):
        # getting the key values from knn
        # if memory store is on cpu, we can do this asynchronously while gpu does some work
        # using knn on key value store
        memKeys = None
        memVals = None

        # computing local attention (self attention in this case)
        locAttention = self.localAttention.forward(query,query,query)

        memAttention = self.memoryAttention.forward()

        finalVal = self.gate(locAttention, memAttention)

        # updating the key value store to include the new key value pairs
        # we can asynchronously add to continue with the model on gpu if memory store is on the cpu



        return finalVal


# defining memory here for faiss
class MemoryStore():
    def __init__(self, memory_size, top_k, dimension):
        self.db = faiss.IndexHNSWFlat(dimension)
        self.memory_size = memory_size
        self.top_k = top_k
        self.dimension = dimension

    def use_gpu(self):
        self.db = faiss.GpuIndexIVFPQ(self.dimension)

    def use_cpu(self):
        self.db = faiss.IndexIVFPQ(self.dimension)

    def clear(self):
        self.db.reset()

    # given a batch of vectors to add to the memory store, we may need to remove older vectors
    # we are not using add_with_ids, so faiss will give sequential ids for adds -> we can use this fact for removal
    def add(self, batch):
        if batch.shape[0] + self.db.ntotal > self.memory_size:
            # we need to remove some old vectors from the memory store
            num_to_remove = batch.shape[0] + self.db.ntotal - self.memory_size
            # removing from beginning of index
            self.db.remove_ids(np.arange(num_to_remove))
        self.db.add(batch)

    # performs the knn search using a query batch
    def knn_search(self, query):
        return self.db.search(query,self.top_k)
