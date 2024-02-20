import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.quantization.quantize_fx import prepare_fx, convert_fx

model_path = "/mnt/New/Data/Vbox_SF/HuggingFaceLocal/"
model_name = "flan-t5-small"
model_folder = model_path + model_name

# Load the pretrained FLAN-T5 model
model = T5ForConditionalGeneration.from_pretrained(model_folder)

# Prepare the model for quantization

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_folder)

# Example input text
input_text = "Translate English to French: How are you?"

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Print tokenized input
print("Tokenized input:", input_ids)

# Use the tokenized input as dummy_input_ids
dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]]) 
qconfig_mapping = {'': torch.quantization.default_dynamic_qconfig}
prepared_model = prepare_fx(model, qconfig_mapping, example_inputs=[dummy_input_ids])

# # Convert the model to AWQ format
# quantized_model = convert_fx(prepared_model, quantization="awq")

# # Save the quantized model
# quantized_model.save_pretrained(model_name + "_AWQ")

# print("Quantized model saved.")




