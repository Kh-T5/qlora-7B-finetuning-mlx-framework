from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import (
    MODEL_NAME,
    MAX_LENGTH,
    Dataset_dolly,
    tokenized_ds_path,
    SEED,
    split_val,
)


### Init tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Using same token for padding and delimitting end of sentence
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id


def preprocess_batch(batch):
    """
    Tokenizes the batch using AutoTokenizer for mistral 7b from huggingface transformers

    Input  : DOLLY 15k batch imported using huggingface datasets in the form
             len(batch) * {"instruction": str, "context": str, "response": str}

    Output : dict {
             "input_ids": list of int representing tokens of instruction, context and response psot concatenation
             "labels": list of ints, allows the model to differentiate between response (what to predict) from the instrution and context,
             "attention_mask": list of int, mask differentiating between pad tokens and real tokens
             }
    """
    instructions = batch["instruction"]
    contexts = batch["context"]
    responses = batch["response"]

    prompts = []  # Concat instruction and context when it is available
    for instr, ctx in zip(instructions, contexts):
        if ctx:
            prompt = f"Instruction: {instr}\nContext: {ctx}\nResponse:"
        else:
            prompt = f"Instruction: {instr}\nResponse:"
        prompts.append(prompt)

    # Tokenize promt and response
    prompt_enc = tokenizer(
        prompts,
        add_special_tokens=False,
    )
    response_enc = tokenizer(
        responses,
        add_special_tokens=False,
    )

    all_input_ids = []
    all_labels = []
    all_attention_masks = []

    for prompt_ids, response_ids in zip(
        prompt_enc["input_ids"], response_enc["input_ids"]
    ):
        ids = (
            prompt_ids + response_ids + [eos_id]
        )  ### Concat prompt, response and end of sentense token for model input
        labels = (
            [-100] * len(prompt_ids) + response_ids + [eos_id]
        )  ### labels to keep track of ids belonging to prompt and the ones belonging to response / special tokens

        # Truncation at MAX_LENGTH
        ids = ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

        pad_len = MAX_LENGTH - len(ids)
        if pad_len > 0:  # When possible fill in remaining space with pad tokens
            ids = ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len

        attention_mask = [1] * (MAX_LENGTH - pad_len) + [
            0
        ] * pad_len  # Mask for padding tokens

        all_input_ids.append(ids)
        all_labels.append(labels)
        all_attention_masks.append(attention_mask)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    }


if __name__ == "__main__":

    print("Loading dataset...")
    dolly_ds = load_dataset(Dataset_dolly)["train"]

    ### Tokenizing the dataset
    print("Tokenizing Dolly dataset...")
    tokenized_ds = dolly_ds.map(preprocess_batch, batched=True)
    print("Splitting...")
    split_ds = tokenized_ds.train_test_split(
        test_size=split_val,
        seed=SEED,
        shuffle=True,
    )

    ### Saving data using hugging face dataset
    print("Saving tokenized dataset...")
    split_ds.save_to_disk(tokenized_ds_path)
    print("Done.")
