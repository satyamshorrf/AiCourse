from ctransformers import AutoModelForCausalLM

model_path = ("../../Models/models--TheBloke--Llama-2-7B-Chat-GGUF/"
              "snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb")

llm = AutoModelForCausalLM.from_pretrained(model_path,
                                           model_file="llama-2-7b.Q4_K_M.gguf",
                                           model_type="llama", gpu_layers=50)
# User interface
prompt = "Which is the largest country in the world by population?"
print(llm("Question: {prompt} Answer:"))