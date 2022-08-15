# definition for the model of vanilla distillbert, but with an added key value store

from transformers import DistilBertModel
from Modules.ValueStore import MultiMemoryHeadedAttention

def getDistilBertValueStore():
    # firstly getting the vanilla model
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # replacing multihead self attention in layer 3 with the MemoryMultiHeadedAttention
    # layer 3 corresponds to transformer block index 2 ( since indexing starts from 0)
    currentAttentionModule = model.transformer.layer[2].attention
    newAttentionModule = MultiMemoryHeadedAttention(768,768,8,True,512,32,0.1)

    # copying attention parameters
    newAttentionModule.load_state_dict(currentAttentionModule.state_dict())

    model.transformer.layer[2].attention = newAttentionModule

    return model


