from transformers import T5Tokenizer, T5ForConditionalGeneration, GPTQConfig
import torch

model_folder = "/mnt/New/Data/Vbox_SF/HuggingFaceLocal/"
#model_folder = "/home/debbie/Dev/HF Finetuning/finetuned/"
model_name = "flan-t5-large"
model_path = model_folder + model_name
quant_path = model_folder + "-awq"

tokenizer = T5Tokenizer.from_pretrained(model_path)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

# Quantize the model
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map='auto', quantization_config=quantization_config)
### Doesn't work because T5 arch is not supported by GPTQ

# Save quantized model
model.save_pretrained(quant_path)