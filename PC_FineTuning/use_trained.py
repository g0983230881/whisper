# from huggingface_hub import notebook_login
# notebook_login()
# 記得先在 terminal 輸入 huggingface-cli login
# 等價於上方兩行登入的程式

# pip install gradio transformers

from transformers import pipeline
import gradio as gr

pipe = pipeline(model="HuangJordan/whisper-small-chinese")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    print("Received file path:", audio)
    if audio is None:
        return "No file received. Please try again."
    
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Record or Upload"),
    outputs="text",
    title="Whisper-small-chinese",
    description="Realtime demo for chinese speech recognition using a fine-tuned Whisper small model.",
)

iface.launch(share=True)