# Original code from https://huggingface.co/blog/fine-tune-whisper
# Modified by Blaz Kovacic for training on Artur 1.0 dataset

# Script usage:
#   python3 train.py demo                (this runs first training round)
#   python3 train.py demo resume #steps  (this resumes from any previous training rounds, and trains until #steps is reached)
#   python3 train.py base                (this runs first training round)
#   python3 train.py base resume #steps  (this resumes from any previous training rounds, and trains until #steps is reached)
#   (similar for baselora, tiny, tinylora)

print("Importing libraries")
import sys
import os
import string

if len(sys.argv) != 2 and len(sys.argv) != 4:
    print("Error! Run script as {} [traintype], where traintype can be one of [demo, base, baselora, tiny, tinylora]. Also, when resuming, specify steps.".format(sys.argv[0]))
    exit(0)

train_type = sys.argv[1]
print("Train type = {}".format(train_type))

if len(sys.argv) == 4:
    if sys.argv[2] == "resume":
        resume_training = True
        resume_train_steps = int(sys.argv[3])
        print("Resuming training from last checkpoint!")
    else:
        print("Error, received wrong second parameter - it can only be 'resume'")
        exit(0)
else:
    resume_training = False
    print("Training for the first time.")

# Set separate cache directories for parallel script execution!
if train_type == "demo":
    local_cache_dir = "/mnt/cache_demo"
elif train_type == "demolora":
    local_cache_dir = "/mnt/cache_demolora"
elif train_type == "base":
    local_cache_dir = "/mnt/cache_base"
elif train_type == "baselora":
    local_cache_dir = "/mnt/cache_baselora"
elif train_type == "tiny":
    local_cache_dir = "/mnt/cache_tiny"
elif train_type == "tinylora":
    local_cache_dir = "/mnt/cache_tinylora"
else:
    local_cache_dir = "/mnt/cache"

os.environ["HF_HOME"] = local_cache_dir
os.environ["TRANSFORMERS_CACHE"] = local_cache_dir
os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # <<--- selects GPU number 0 (A100 Singularity issue: https://discuss.pytorch.org/t/cannot-find-gpu-on-a100-in-singularity/172936)

import datasets
from datasets import load_dataset, DatasetDict
from huggingface_hub import login, snapshot_download #, list_repo_files, hf_hub_download
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import pandas as pd
import gc
import numpy as np

#import librosa
import tokenize
import re
import random
from datasets.arrow_writer import ArrowWriter

import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from peft import AdaLoraModel, AdaLoraConfig

hf_token = "YOUR_TOKEN_HERE"

# 300000 samples [train+valid]; 5,92 samples/second
# 8500 samples per epoch ==> 6 epochs = 51000 samples

# HYPERPARAMETERS
if train_type == "demo":
    learning_rate = 2.5e-5  # (recommended for base Whisper model by https://github.com/vasistalodagala/whisper-finetune)
    train_batch_size = 16   # (set to 16 if only 8GB VRAM)
    grad_accumulation_steps = 2 # (set to 2 if only 8GB VRAM)
    max_steps = 700
    save_steps = 100
    eval_steps = 100
    eval_batch_size = 16    # (could try 32 as well)
    whisper_model_string = "whisper-base"
    model_repo_dir = "/mnt/whisper-base-sl-artur-full-demo" # saved model repository
    model_hub_dir = "blko/whisper-base-sl-artur-full-demo"
    #dataset_save_path = "/mnt/dataset/artur1_0_demo"
    dataset_save_path = "/mnt2/artur1_0"
elif train_type == "demolora":
    learning_rate = 2.0e-4
    train_batch_size = 16   # (set to 16 if only 8GB VRAM)
    grad_accumulation_steps = 1 # (set to 2 if only 8GB VRAM)
    max_steps = 5000
    save_steps = 100
    eval_steps = 100
    eval_batch_size = 16    # (could try 32 as well)
    whisper_model_string = "whisper-base"
    model_repo_dir = "/mnt/whisper-base-sl-artur-lora-demo" # saved model repository
    model_hub_dir = "blko/whisper-base-sl-artur-lora-demo"
    dataset_save_path = "/mnt/dataset/artur1_0_demo"
elif train_type == "base":
    learning_rate = 2.5e-5  # (recommended for base Whisper model by https://github.com/vasistalodagala/whisper-finetune)
    train_batch_size = 32   # (set to 16 if only 8GB VRAM)
    grad_accumulation_steps = 1 # (set to 2 if only 8GB VRAM)
    max_steps = 51000#28125       # (with train_batch_size=32, this will pass 32*28125 = 900000 samples, i.e. 3 epochs because dataset has 300k samples)
    save_steps = 4000        # (each checkpoint requires ~1GB disk space, and ~300MB on HuggingFace hub)
    eval_steps = 4000
    eval_batch_size = 16    # (could try 32 as well)
    whisper_model_string = "whisper-base"
    model_repo_dir = "/mnt/whisper-base-sl-artur-full-ft" # saved model repository
    model_hub_dir = "blko/whisper-base-sl-artur-full-ft"
    dataset_save_path = "/mnt/dataset/artur1_0"
elif train_type == "baselora":
    learning_rate = 1.5e-4 #2.5e-5  # (recommended for base Whisper model by https://github.com/vasistalodagala/whisper-finetune)
    train_batch_size = 32   # (set to 16 if only 8GB VRAM)
    grad_accumulation_steps = 1 # (set to 2 if only 8GB VRAM)
    max_steps = 51000#28125       # (with train_batch_size=32, this will pass 32*28125 = 900000 samples, i.e. 3 epochs because dataset has 300k samples)
    save_steps = 4000        # we end up with 28 checkpoints
    eval_steps = 4000
    eval_batch_size = 16    # (could try 32 as well)
    whisper_model_string = "whisper-base"
    model_repo_dir = "/mnt/whisper-base-sl-artur-lora-ft" # saved model repository
    model_hub_dir = "blko/whisper-base-sl-artur-lora-ft"
    dataset_save_path = "/mnt/dataset/artur1_0"
elif train_type == "tiny":
    learning_rate = 3.75e-5  # (recommended for tiny Whisper model by https://github.com/vasistalodagala/whisper-finetune)
    train_batch_size = 32   # (set to 16 if only 8GB VRAM)
    grad_accumulation_steps = 1 # (set to 2 if only 8GB VRAM)
    max_steps = 51000#28125       # (with train_batch_size=32, this will pass 32*28125 = 900000 samples, i.e. 3 epochs because dataset has 300k samples)
    save_steps = 4000        # we end up with 28 checkpoints
    eval_steps = 4000
    eval_batch_size = 16    # (could try 32 as well)
    whisper_model_string = "whisper-tiny"
    model_repo_dir = "/mnt/whisper-tiny-sl-artur-full-ft" # saved model repository
    model_hub_dir = "blko/whisper-tiny-sl-artur-full-ft"
    dataset_save_path = "/mnt/dataset/artur1_0"
elif train_type == "tinylora":
    learning_rate = 1.5e-4 #3.75e-5  # (recommended for tiny Whisper model by https://github.com/vasistalodagala/whisper-finetune)
    train_batch_size = 32   # (set to 16 if only 8GB VRAM)
    grad_accumulation_steps = 1 # (set to 2 if only 8GB VRAM)
    max_steps = 51000#28125       # (with train_batch_size=32, this will pass 32*28125 = 900000 samples, i.e. 3 epochs because dataset has 300k samples)
    save_steps = 4000        # we end up with 28 checkpoints
    eval_steps = 4000
    eval_batch_size = 16    # (could try 32 as well)
    whisper_model_string = "whisper-tiny"
    model_repo_dir = "/mnt/whisper-tiny-sl-artur-lora-ft" # saved model repository
    model_hub_dir = "blko/whisper-tiny-sl-artur-lora-ft"
    dataset_save_path = "/mnt/dataset/artur1_0"

# IF WE RESUME TRAINING, OVERRIDE "max_steps"
if resume_training == True:
    max_steps = resume_train_steps

login(hf_token, add_to_git_credential=True, write_permission=True)

if resume_training == False:
    tokenizer = WhisperTokenizer.from_pretrained("openai/" + whisper_model_string, language="Slovenian", task="transcribe", predict_timestamps=True)
    processor = WhisperProcessor.from_pretrained("openai/" + whisper_model_string, language="Slovenian", task="transcribe", predict_timestamps=True)
else:
    tokenizer = WhisperTokenizer.from_pretrained(model_hub_dir, language="Slovenian", task="transcribe", predict_timestamps=True)
    processor = WhisperProcessor.from_pretrained(model_hub_dir, language="Slovenian", task="transcribe", predict_timestamps=True)

train_dataset_save_path = os.path.join(dataset_save_path, "train")
valid_dataset_save_path = os.path.join(dataset_save_path, "valid")
test_dataset_save_path = os.path.join(dataset_save_path, "test")

train_path_list = os.listdir(train_dataset_save_path)
valid_path_list = os.listdir(valid_dataset_save_path)
test_path_list = os.listdir(test_dataset_save_path)

# shuffle .parquet files (train data), so that when training multiple times, there's some difference
random.shuffle(train_path_list)
random.shuffle(valid_path_list)
random.shuffle(test_path_list)

train_ds = datasets.load_dataset(train_dataset_save_path, data_files=train_path_list, streaming=True, split="train")
valid_ds = datasets.load_dataset(valid_dataset_save_path, data_files=valid_path_list, streaming=True, split="train")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        #print(type(features), features[0].keys())
        
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")

def normalize_text(text):   # suggested by AI
    # 1. Lowercasing
    text = text.lower()
    # 2. Remove words that start with hashtags
    text = re.sub(r'#\w+', '', text)
    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 4. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Return the normalized string (not tokenized)
    return text

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # (NEW) text gets normalized
    norm_pred_str = [normalize_text(tmp_str) for tmp_str in pred_str]
    norm_label_str = [normalize_text(tmp_str) for tmp_str in label_str]
    
    filtered_pred_str = []
    filtered_label_str = []
    for i in range(len(norm_label_str)):
        if norm_pred_str[i].strip() and norm_label_str[i].strip():
            filtered_pred_str.append(norm_pred_str[i])
            filtered_label_str.append(norm_label_str[i])
    
    # Finds all indices of elements with empty strings and remove elements of those indices
    indices_to_remove = [i for i, x in enumerate(filtered_label_str) if x==""]
    mod_true_text = [x for i, x in enumerate(filtered_label_str) if i not in indices_to_remove]
    mod_pred_text = [x for i, x in enumerate(filtered_pred_str) if i not in indices_to_remove]
    if len(mod_true_text) == 0 or len(mod_pred_text) == 0:
        return {"wer": 100.0}
    
    if mod_pred_text and mod_true_text:        
        wer = 100 * metric.compute(predictions=mod_pred_text, references=mod_true_text)
    else:
        wer = 100.0
    
    return {"wer": wer}

if resume_training == False:
    model = WhisperForConditionalGeneration.from_pretrained("openai/" + whisper_model_string)
else:
    model = WhisperForConditionalGeneration.from_pretrained(model_hub_dir)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = None #<------------- modified this because of an error 

if train_type == "baselora" or train_type == "tinylora" or train_type == "demolora":
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_repo_dir,  # change to a repo name of your choice
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=500,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        ignore_data_skip=True,  # <--- this must be set when "resuming" training
        #save_safetensors=False,  # this prevents error "There were missing keys in the checkpoint model loaded: ['proj_out.weight']." when resuming training
        remove_unused_columns=False, # required by LoRA
        label_names=["labels"]       # required by LoRA
    )
else:
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_repo_dir,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=500,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=eval_batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        ignore_data_skip=True       # <--- this must be set when "resuming" training
        #save_safetensors=False  # this prevents error "There were missing keys in the checkpoint model loaded: ['proj_out.weight']." when resuming training
    )

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

if resume_training == False:
    trainer.train()
else:
    trainer.train(resume_from_checkpoint=True)

if train_type == "demo":
    kwargs = {
        "dataset_tags": "Artur-1-0-demo",
        "dataset": "Artur 1.0 Demo Dataset",  # a 'pretty' name for the training dataset
        "dataset_args": "config: sl, split: test",
        "language": "sl",
        "model_name": "Whisper Base Slo Artur - Full Demo",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-base",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }
elif train_type == "demolora":
    kwargs = {
        "dataset_tags": "Artur-1-0-demo",
        "dataset": "Artur 1.0 Demo Dataset",  # a 'pretty' name for the training dataset
        "dataset_args": "config: sl, split: test",
        "language": "sl",
        "model_name": "Whisper Base Slo Artur - LoRA Demo",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-base",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }
elif train_type == "base":
    kwargs = {
        "dataset_tags": "Artur-1-0-full",
        "dataset": "Artur 1.0 Full Dataset",  # a 'pretty' name for the training dataset
        "dataset_args": "config: sl, split: test",
        "language": "sl",
        "model_name": "Whisper Base Slo Artur - Full FT",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-base",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }
elif train_type == "tiny":
    kwargs = {
        "dataset_tags": "Artur-1-0-full",
        "dataset": "Artur 1.0 Full Dataset",  # a 'pretty' name for the training dataset
        "dataset_args": "config: sl, split: test",
        "language": "sl",
        "model_name": "Whisper Tiny Slo Artur - Full FT",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-tiny",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }
elif train_type == "baselora":
    kwargs = {
        "dataset_tags": "Artur-1-0-full",
        "dataset": "Artur 1.0 Full Dataset",  # a 'pretty' name for the training dataset
        "dataset_args": "config: sl, split: test",
        "language": "sl",
        "model_name": "Whisper Base Slo Artur - LoRA FT",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-base",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }
elif train_type == "tinylora":
    kwargs = {
        "dataset_tags": "Artur-1-0-full",
        "dataset": "Artur 1.0 Full Dataset",  # a 'pretty' name for the training dataset
        "dataset_args": "config: sl, split: test",
        "language": "sl",
        "model_name": "Whisper Tiny Slo Artur - LoRA FT",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-tiny",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }

trainer.push_to_hub(**kwargs)

if train_type == "baselora":
    model = model.merge_and_unload()
    model.save_pretrained("whisper-base-sl-artur-lora-ft")
    model.push_to_hub("whisper-base-sl-artur-lora-ft")
elif train_type == "tinylora":
    model = model.merge_and_unload()
    model.save_pretrained("whisper-tiny-sl-artur-lora-ft")
    model.push_to_hub("whisper-tiny-sl-artur-lora-ft")
elif train_type == "demolora":
    model = model.merge_and_unload()
    model.save_pretrained("whisper-base-sl-artur-lora-demo")
    model.push_to_hub("whisper-base-sl-artur-lora-demo")

print("Done training {}!".format(train_type))
