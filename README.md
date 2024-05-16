# whisper檔案介紹  
## output/  
為模型辨識出來的輸出檔案, 預設內含json、srt、tsv、txt、vtt五種檔案格式  
## PC_FineTuning/  
為各個模型訓練用的程式, 僅透過讀取網路上dataset訓練  
## wav_folder/  
音源資料夾(訓練/測試集)  
## whisper-main/  
官方源程式碼, 參考用  
### demo.py  
用官方模型推論  
### Fine_Tuning.py(ipynb) 
完整的訓練程式碼  
### metadata.csv 
當中處存兩個欄位, file_name、transcription, 給存取本地音檔程式用  
### upload_wav_dataset.py  
存取本地音檔做訓練的完整程式碼  
### upload_wav_dataset_to_huggingface.py  
將本地音檔push到huggingface社群中  
### use_trained_model_gradio  
建立gradio網頁使用已訓練好的模型, 推論輸入音檔
