#!/usr/bin/env python
# coding: utf-8

# In[11]:


# from huggingface_hub import notebook_login
# notebook_login()
# 記得先在 terminal 輸入 huggingface-cli login
# 並輸入 huggingface 帳戶的 Access Tokens 若無則新增一個, 記得權限選擇 Write
# 等價於上方登入 huggingface_hub 的程式


# In[12]:
# dataset: mozilla-foundation/common_voice_11_0, zh-TW
#          google/fleurs, cmn_hans_cn

from datasets import load_dataset, DatasetDict

datasets = DatasetDict()

datasets["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-TW", split="train+validation", token='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', trust_remote_code=True)
datasets["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-TW", split="test", token='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', trust_remote_code=True)

print(datasets)


# In[13]:

# datasets = datasets.remove_columns(["id", "num_samples", "path", "transcription", "gender", "lang_id", "language", "lang_group_id"])
datasets = datasets.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

print(datasets)


# In[14]:


from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")


# In[15]:


from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="chinese", task="transcribe")


# In[16]:


from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="chinese", task="transcribe")


# In[17]:


print(datasets["train"][0])


# In[18]:

# load and resample audio data from 48kHz to 16kHz
from datasets import Audio
datasets = datasets.cast_column("audio", Audio(sampling_rate=16000))


# In[19]:


print(datasets["train"][0])


# In[20]:


def prepare_dataset(batch):
    
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    # batch["labels"] = tokenizer(batch["sentence"]).input_ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# In[21]:


datasets = datasets.map(prepare_dataset, remove_columns=datasets.column_names["train"], num_proc=2)


# In[22]:


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")


# In[23]:


model.generation_config.language = "chinese"
model.generation_config.task = "transcribe"

# model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.suppress_tokens = []

# In[24]:


import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt", padding=True)

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# In[25]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# In[26]:


import evaluate

metric = evaluate.load("cer")


# In[27]:


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


# In[39]:


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="HuangJordan/whisper-medium-chinese-cer",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=200,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    push_to_hub=True,
)


# In[40]:


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


# In[41]:


processor.save_pretrained(training_args.output_dir)


# In[38]:


trainer.train()


# In[ ]:


kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: chinese, split: test",
    "language": "zh",
    "model_name": "Whisper medium mozilla-foundation/common_voice_11_0 - Huang Jordan",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-medium",
    "tasks": "automatic-speech-recognition",
    # language跟config不一樣
}


# In[ ]:


trainer.push_to_hub(**kwargs)


# In[ ]:




