import torch
from transformers import T5ForConditionalGeneration

model_path = "C:\\Users\\JV Roig\\Dev\\HuggingFaceLocal\\"
model_name = "flan-t5-small"
model_folder = model_path + model_name

# Load the pretrained T5 model
model_name = "t5-small"  # You can use any T5 variant here
model = T5ForConditionalGeneration.from_pretrained(model_folder)

# Quantize the model using AWQ
quantized_model = torch.quantization.quantize_dynamic(
    model,  # Original model
    {torch.nn.Linear},  # Desired types to quantize
    dtype=torch.float16,  # Target data type for quantized weights and activations
    quant_type='awq',  # Use AWQ for quantization
)

# Save the quantized model
quantized_model.save_pretrained(model_name + "_AWQ")

print("Quantized model saved.")