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
# from transformers import WhisperFeatureExtractor,WhisperTokenizer, WhisperProcessor
# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="chinese", task="transcribe")
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="chinese", task="transcribe")


# 調用 batch["audio"] 加載音頻並轉換成梅爾頻譜, 透過path指定檔案並新增轉錄欄位
# def prepare_data(batch):
#     audio = batch['audio']
#     file_name = audio["path"].split('/')[-1]
#     transcription = transcription_dict.get(file_name, "未找到轉錄")
    
#     batch["audio_feature"] = audio
#     batch['sentences'] = transcription
#     return batch

def prepare_data(batch):
    audio = batch['audio']
    file_name = audio["path"].split('/')[-1]
    transcription = transcription_dict.get(file_name, "未找到轉錄")
    
    batch["audio_feature"] = audio
    batch['labels'] = transcription
    return batch

dataset_folder = dataset_folder.map(prepare_data, remove_columns=dataset_folder.column_names["train"])
# print(dataset_folder)
# for i in range(len(dataset_folder["train"])):
#     print(dataset_folder["train"][i]["audio_feature"], dataset_folder["train"][i]['labels'])

# dataset_folder.push_to_hub("HuangJordan/Whisper_Dataset")
# load from huggingface_dataset to preprocess these files
# datasets = load_dataset("HuangJordan/Whisper_Dataset", token='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', trust_remote_code=True)
# print(datasets['train'][0])

