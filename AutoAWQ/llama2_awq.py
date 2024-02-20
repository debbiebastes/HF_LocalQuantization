from awq import AutoAWQForCausalLM
from transformers import LlamaTokenizer

model_folder = "/mnt/New/Data/Vbox_SF/llama/"
#model_folder = "/home/debbie/Dev/HF Finetuning/finetuned/"
model_name = "llama-2-7b-chat"
model = model_folder + model_name

quant_path = model_name + "-awq"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
tokenizer = LlamaTokenizer.from_pretrained(model, local_files_only=True)
model = AutoAWQForCausalLM.from_pretrained(model)


# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)