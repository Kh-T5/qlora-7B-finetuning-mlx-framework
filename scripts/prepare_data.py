from datasets import load_dataset
from transformers import AutoTokenizer

from mistral_qlora.config import Paths, TrainConfig
from mistral_qlora.constants import DATASET_NAME, IGNORE_INDEX, MODEL_NAME

TRAIN_CONFIG = TrainConfig()
PATHS = Paths()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

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

    prompts = []
    for instr, ctx in zip(instructions, contexts, strict=True):
        if ctx:
            prompt = f"Instruction: {instr}\nContext: {ctx}\nResponse:"
        else:
            prompt = f"Instruction: {instr}\nResponse:"
        prompts.append(prompt)

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
        prompt_enc["input_ids"], response_enc["input_ids"], strict=True
    ):
        ids = prompt_ids + response_ids + [eos_id]
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids + [eos_id]

        ids = ids[: TRAIN_CONFIG.max_length]
        labels = labels[: TRAIN_CONFIG.max_length]

        pad_len = TRAIN_CONFIG.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [pad_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len

        attention_mask = [1] * (TRAIN_CONFIG.max_length - pad_len) + [0] * pad_len

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
    dolly_ds = load_dataset(DATASET_NAME)["train"]

    print("Tokenizing Dolly dataset...")
    tokenized_ds = dolly_ds.map(preprocess_batch, batched=True)
    print("Splitting...")
    split_ds = tokenized_ds.train_test_split(
        test_size=TRAIN_CONFIG.val_split,
        seed=TRAIN_CONFIG.seed,
        shuffle=True,
    )

    print("Saving tokenized dataset...")
    split_ds.save_to_disk(str(PATHS.tokenized_dataset))
    print("Done.")
