# preparing datasets for training and evaluation
# these functions transform a single example, and then we will use huggingface dataset "map" for batching
# huggingface has a lot of examples on data preparation for various nlp tasks using their dataset library
#             and tokenizers, so I am referring to that while writing this code

from datasets import load_dataset
from transformers import DistilBertTokenizerFast
import numpy as np

split = 'test'
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

# openbookqa is a multiple choice based language understanding task
# openbookqa has 'train', 'validation', and 'test' splits
def loadOpenbookQA(tokenizer):
    raw_data = load_dataset('openbookqa')
    if split == 'train':
        processed = raw_data['train'].map(
            prepareOpenbookQA,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=raw_data["train"].column_names,
        )
    elif split == 'validation':
        processed = raw_data['validation'].map(
            prepareOpenbookQA,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=raw_data["validation"].column_names,
        )
    else:
        processed = raw_data['test'].map(
            prepareOpenbookQA,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=raw_data["test"].column_names,
        )
    return processed


# per-example preprocessing for openbookqa (to be used with batch processing in huggingface dataset)
def prepareOpenbookQA(examples,tokenizer):
    # we need to duplicate question stem 4 times for each choice
    question_texts = examples['question_stem']
    repeated_questions = [[question] * 4 for question in question_texts]
    # tokenizing premises with the choices
    choices = examples['choices']
    choice_texts = [choice['text'] for choice in choices]
    # need to flatten everything before tokenizing
    repeated_questions = np.array(repeated_questions).flatten().tolist()
    choice_texts = np.array(choice_texts).flatten().tolist()
    tokenized = tokenizer(repeated_questions,choice_texts,
                          truncation=True,
                          max_length=tokenizer.max_len_sentences_pair)

    # need to return groups of 4 basically with the ground truth label
    # we use char arithmetic here for the correct label index, since 'A'-'A' -> 0 , 'B'-'A'->1 and so on
    gold_labels = [ord(choice_letter) - ord('A') for choice_letter in examples['answerKey']]
    # grouping each batch of 4 encodings
    question_batch = {k: [v[i:i+4] for i in range(0,len(v),4)] for k,v in tokenized.items()}
    # adding on gold indices
    question_batch['gold_labels'] = gold_labels
    return question_batch

# common sense qa is a 5 question multiple choice dataset
# important note is that test set has no answer key, so we will just use A for a dummy variable
# splits are 'validation', 'train', and 'test'
# preprocessing is similar to openbookqa
def loadCommonSenseQA(tokenizer):
    raw_data = load_dataset('commonsense_qa')
    if split == 'train':
        processed = raw_data['train'].map(
            prepareCommonsenseQA,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=raw_data["train"].column_names,
        )
    elif split == 'validation':
        processed = raw_data['validation'].map(
            prepareCommonsenseQA,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=raw_data["validation"].column_names,
        )
    else:
        processed = raw_data['test'].map(
            prepareCommonsenseQA,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=raw_data["test"].column_names,
        )
    return processed

# preparing commonsense QA
def prepareCommonsenseQA(examples,tokenizer):
    # we need to duplicate question stem 5 times for each choice
    question_texts = examples['question']
    repeated_questions = [[question] * 5 for question in question_texts]
    # tokenizing premises with the choices
    choices = examples['choices']
    choice_texts = [choice['text'] for choice in choices]
    # need to flatten everything before tokenizing
    repeated_questions = np.array(repeated_questions).flatten().tolist()
    choice_texts = np.array(choice_texts).flatten().tolist()
    tokenized = tokenizer(repeated_questions, choice_texts,
                          truncation=True,
                          max_length=tokenizer.max_len_sentences_pair)

    # need to return groups of 5 basically with the ground truth label
    # we use char arithmetic here for the correct label index, since 'A'-'A' -> 0 , 'B'-'A'->1 and so on
    if split == 'test':
        # then we have the test split, where there is no answer key
        gold_labels = [0 for i in range(len(examples['question']))]
    else:
        gold_labels = [ord(choice_letter) - ord('A') for choice_letter in examples['answerKey']]
    # grouping each batch of 5 encodings
    question_batch = {k: [v[i:i + 5] for i in range(0, len(v), 5)] for k, v in tokenized.items()}
    # adding on gold indices
    question_batch['gold_labels'] = gold_labels
    return question_batch

# preparing PG-19 language modeling
def preparePG19(data):
    pass

# glue benchmark testing
def loadGlueBenchmark(tokenizer):
    pass

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


