from awq import AutoAWQForCausalLM
from transformers import LlamaTokenizer

#Importing hf_local_config from parent folder takes a bit more work than if it were in the same folder as this script
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from hf_local_config import *

model_name = 'llama-2-7b-chat'
model_id   = model_path+model_name

# Load model
tokenizer = LlamaTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoAWQForCausalLM.from_pretrained(
    model_id
)

# Quantize
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
quant_path = model_name + "-awq"
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')