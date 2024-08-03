from transformers import pipeline
import torch

model_path = ("../../Models/models--openai--whisper-medium/"
              "snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline(task="automatic-speech-recognition",
                model=model_path, chunk_length_s=30, device=device)

out = pipe("../../Resources/test.mp3.flac")["text"]
print(out)