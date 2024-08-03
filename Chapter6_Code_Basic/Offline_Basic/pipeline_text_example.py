from transformers import pipeline

model_path = ("../../Models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/"
              "snapshots/714eb0fa89d2f80546fda750413ed43d93601a13")

# Default model
classifier = pipeline(task="text-classification",
                      model=model_path)

print(classifier(["You are the best", "Get Lost"]))