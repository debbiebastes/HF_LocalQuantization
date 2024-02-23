from transformers import T5Tokenizer, T5ForConditionalGeneration, GPTQConfig
import torch

#Importing hf_local_config from parent folder takes a bit more work than if it were in the same folder as this script
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from hf_local_config import *

model_name = "flan-t5-large"
model_id   = model_path+model_name

tokenizer = T5Tokenizer.from_pretrained(model_path)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

# Quantize the model
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map='auto', quantization_config=quantization_config)
### Doesn't work because T5 arch is not supported by GPTQ

# Save quantized model
quant_path = model_path + "-awq"
model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')