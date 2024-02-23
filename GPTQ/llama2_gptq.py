from transformers import LlamaForCausalLM, LlamaTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch

#Importing hf_local_config from parent folder takes a bit more work than if it were in the same folder as this script
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from hf_local_config import *

model_name = "llama-2-7b-chat"
model_id   = model_path+model_name

# Load model
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto')

#quantize model
quantizer = GPTQQuantizer(bits=4, dataset="c4")
quantized_model = quantizer.quantize_model(model, tokenizer)

# Save quantized model
quant_path = model_path + "-awq"
quantizer.save(quantized_model,quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')