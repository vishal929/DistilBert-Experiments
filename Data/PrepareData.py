# preparing datasets for training and evaluation
# these functions transform a single example, and then we will use huggingface dataset "map" for batching

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

split = 'validation'
stride = 128

# squad has 'train' and 'validation' splits
def loadSquad(tokenizer):
    squad = load_dataset('squad')

    if split == 'train':
        processed = squad['train'].map(
            huggingPrepareSquadTrain,
            batched=True,
            fn_kwargs={'tokenizer':tokenizer},
            remove_columns=squad["validation"].column_names,
        )
    else:
        processed = squad['validation'].map(
            huggingPrepareSquadVal,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=squad["validation"].column_names,
        )
    print(processed)

# huggingface provided qa preparation for squad
def huggingPrepareSquadVal(examples,tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=tokenizer.max_len_sentences_pair,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    #sample_map = list(range(len(inputs['input_ids'])))
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


# huggingface provided qa preparation for squad
def huggingPrepareSquadTrain(examples,tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=tokenizer.max_len_sentences_pair,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    #sample_map = list(range(len(inputs['input_ids'])))
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def prepareOpenbookQA(data):
    pass

# preparing medQA-USMLE
def prepareMedQA(data):
    pass

# preparing commonsense QA
def prepareCommonsenseQA(data):
    pass

# preparing PG-19 language modeling
def preparePG19(data):
    pass

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

tokenized = loadSquad(tokenizer)

'''
validating that answers in squad are uniform
for row in squad:
    if len(squad[0]['answers']['text'])>0:
        answers = squad[0]['answers']['text']
        if len(set(answers)) >1:
            print(row)
'''
# setting pytorch format for eval
'''
squad.set_format()
#squad.set_format(type='torch',columns=['id','title','context','question','answers'])
squadTorch = DataLoader(squad, batch_size = 1)
for batch in squadTorch:
    print(batch['title'])
'''