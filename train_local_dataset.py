from datasets import Dataset, Audio, DatasetDict, load_dataset
import pandas as pd

# 讀取 metadata.csv 並轉換成 dict
transcriptions_df = pd.read_csv("metadata.csv")
transcription_dict = pd.Series(transcriptions_df.transcription.values, index=transcriptions_df.file_name).to_dict()
print(transcription_dict)

# 存取本地音檔資料夾
dataset_folder = load_dataset("audiofolder", data_dir="wav_folder")
dataset_folder = dataset_folder["train"].train_test_split(test_size=0.2)
dataset_folder = dataset_folder.cast_column("audio", Audio(sampling_rate=16000))
print(dataset_folder["train"])
print(dataset_folder["test"])

# 設置 feature_extractor tokenizer processor
from transformers import WhisperFeatureExtractor,WhisperTokenizer, WhisperProcessor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="chinese", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="chinese", task="transcribe")

# 調用 batch["audio"] 加載音頻並轉換成梅爾頻譜, 透過path指定檔案來新增轉錄欄位
def prepare_data(batch):
    audio = batch['audio']
    # print(audio["path"])
    file_name = audio["path"].split('/')[-1]
    transcription = transcription_dict.get(file_name, "未找到轉錄")
    
    # batch["audio_feature_orgin"] = audio
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch['labels'] = tokenizer(transcription).input_ids
    return batch

dataset_folder = dataset_folder.map(prepare_data, remove_columns=dataset_folder.column_names["train"])
# for i in range(len(dataset_folder["train"])):
#     print(dataset_folder["train"][i]["audio_feature_orgin"]["path"], dataset_folder["train"][i]['labels'])
print(dataset_folder['train'])


import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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


import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="HuangJordan/whisper-base-kipt",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_folder["train"],
    eval_dataset=dataset_folder["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()

