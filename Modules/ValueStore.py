# defining modules needed for a value store
# inspiration is from paper Memorizing Transformers: https://arxiv.org/pdf/2203.08913.pdf
# we will use faiss approximate k-nn

import torch
import faiss
import os
import numpy as np

# external file for indexing that needs to spill to disk
db_file_path = os.path.join(os.getcwd(), "Resources", "value_store")


# this represents the gating module that "chooses" between model values and memory store values
class GatedLocalValueAttention(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        # scalar gating parameter called gamma
        # so, we have a parameter for each head , since each head does this gating operation
        self.gamma = torch.tensor(num_heads)

    def forward(self, localAttention, keyValueStoreAttention):
        # simply using the parameter to scale local attention and keyValueStoreAttention
        # this is just "choosing" between using the transformer attention vs key value store knn attention
        # via element wise multiplication
        gate = torch.sigmoid(self.gamma)
        return localAttention * gate + keyValueStoreAttention * (1 - gate)


# defines multi headed attention, but using heads with memory
class MultiMemoryHeadedAttention(torch.nn.Module):
    """
        num_heads: num heads to split for multiheaded attention
        bias: whether we are using a bias or not for linear layers
        memory_size: max number of key,value pairs allowed in caches
        top_k: number of nearest neighbors to return from caches during forward
        out_features: output size to be concatenated for each head -> SHOULD BE DIVISIBLE BY num_heads!!!
    """

    def __init__(self, in_features, out_features, num_heads, bias, memory_size, top_k, dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.top_k = top_k

        # memory modules for each head
        # in_features * 2 for the key,value pair
        self.head_memories = [MemoryStore(self.memory_size, self.top_k, self.in_features * 2) for i in range(num_heads)]

        # actual layers needed
        self.dropout = torch.nn.Dropout(dropout)

        self.q_lin = torch.nn.Linear(in_features, out_features, bias)
        self.k_lin = torch.nn.Linear(in_features, out_features, bias)
        self.v_lin = torch.nn.Linear(in_features, out_features, bias)

        self.softmax = torch.nn.Softmax()

        self.mem_local_gating = GatedLocalValueAttention(out_features // num_heads)

        self.out_lin = torch.nn.Linear(out_features, out_features, bias)

    def forward(self, x):
        head_dim = self.out_features // self.num_heads
        x = self.dropout(x)

        # linear layers for query key and value
        query = self.q_lin(x)

        # splitting query for number of heads
        query = query.view(query.shape[0],-1,self.num_heads,head_dim)

        # starting thread to gather top_k matches from cache for key and value pair
        mem_pairs = torch.stack([self.head_memories[i].knn_search(query) for i in range(self.num_heads)])

        # since we get pairs, we need to split the pairs  into key and values
        mem_key, mem_value = torch.split(mem_pairs, dim=-1)

        key = self.q_lin(x)
        value = self.q_lin(x)

        # splitting key and value for number of heads
        key = key.view(key.shape[0],-1,self.num_heads,head_dim)
        value = value.view(value.shape[0],-1,self.num_heads,head_dim)


        # starting thread to insert new key,value pairs into the cache
        for i, vec in enumerate(torch.concat([key, value], dim=-1)):
            self.head_memories[i].add(vec)

        # attention operation for transformer output
        attention = self.softmax((query * key.transpose()) / torch.sqrt(head_dim)) * value

        # attention operation for memory output
        mem_attention = self.softmax((query * mem_key.transpose()) / torch.sqrt(head_dim)) \
                        * mem_value

        # gating to "choose" between transformer attention and memory attention for each head
        attention = self.mem_local_gating(attention, mem_attention)

        # concatenation for all heads
        concat = torch.stack(attention, -1)

        # output linear layer
        return self.out_lin(concat)


# this is different from the above class in that we are caching Layer-wise instead of head-wise
# purpose of this method is to perform both local attention and memory knn attention
# then use the gating module to weight both memory and local representations
# intended to be the last major layer of a transformer
class KeyValueStoreLayer(torch.nn.Module):
    """
        memory_size: number of key,value token pairs possible
        top_k: number of candidates to retrieve in k-nn algorithm
    """

    def __init__(self, memory_size, top_k, in_dim, out_dim, dropout):
        super().__init__()
        # initializing the key value store -> in_dim *2 for memory dimension since we are storing key,value pairs
        self.memory = MemoryStore(memory_size, top_k, in_dim * 2)
        self.localAttention = torch.nn.MultiheadAttention(embed_dim=in_dim, dropout=dropout)
        self.memoryAttention = torch.nn.MultiheadAttention(embed_dim=in_dim, dropout=dropout)
        self.gate = GatedLocalValueAttention(out_dim)

    def forward(self, x):
        # getting the key values from knn
        # if memory store is on cpu, we can do this asynchronously while gpu does some work
        # using knn on key value store
        memKeyValPairs = self.memory.top_k(x)
        # splitting the retrieved pairs for computation
        memKey = None
        memVal = None

        # computing local attention (self attention in this case)
        locAttention = self.localAttention.forward(x, x, x)

        memAttention = self.memoryAttention.forward(x, memKey, memVal)

        finalVal = self.gate(locAttention, memAttention)

        # updating the key value store to include the new key value pairs
        # we can asynchronously add to continue with the model on gpu if memory store is on the cpu
        self.memory.add(finalVal)

        return finalVal


# defining memory here for faiss
class MemoryStore():
    def __init__(self, memory_size, top_k, dimension):
        #self.db = faiss.IndexHNSWFlat(dimension)
        self.db = None
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
        return self.db.search(query, self.top_k)
