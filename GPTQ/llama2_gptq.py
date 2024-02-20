from transformers import LlamaForCausalLM, LlamaTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch

model_folder = "/mnt/New/Data/Vbox_SF/llama/"
#model_folder = "/home/debbie/Dev/HF Finetuning/finetuned/"
model_name = "llama-2-7b-chat"
model_path = model_folder + model_name
quant_path = model_folder + "-awq"

# Load model
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto')

#quantize model
quantizer = GPTQQuantizer(bits=4, dataset="c4")
quantized_model = quantizer.quantize_model(model, tokenizer)

# Save quantized model
quantizer.save(quantized_model,quant_path)