import torch
from transformers import T5ForConditionalGeneration


model_path = "/mnt/New/Data/Vbox_SF/HuggingFaceLocal/"
model_name = "flan-t5-small"
model_folder = model_path + model_name

# Load the pretrained FLAN-T5 model
model = T5ForConditionalGeneration.from_pretrained(model_folder)

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model,  # Original model
    {torch.nn.Linear},  # Desired types to quantize
    dtype=torch.qint8  # Target data type for quantized weights and activations
)

# Save the quantized model
quantized_model.save_pretrained(model_name + "_int8")

