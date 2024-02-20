import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = "C:\\Users\\JV Roig\\Dev\\HuggingFaceLocal\\"
model_name = "flan-t5-small"
model_folder = model_path + model_name

# Load the pretrained T5 model
model_name = "t5-small"  # You can use any T5 variant here
model = T5ForConditionalGeneration.from_pretrained(model_folder)
tokenizer = T5Tokenizer.from_pretrained(model_folder)

# Sample input text
input_text = "Translate English to French: How are you?"

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Perform model inference
outputs = model.generate(input_ids)

# Decode the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Translated text:", output_text)


# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model,  # Original model
    dtype=torch.qint8  # Target data type for quantized weights and activations
)

# Save the quantized model
quantized_model.save_pretrained(model_name + "_qint8")

print("Quantized model saved.")