# Step 1: Install necessary packages

# Step 2: Download and convert the model
import torch
from transformers import AutoTokenizer, BlenderbotForConditionalGeneration
from nncf import NNCFConfig
from nncf.quantization import quantize, QuantizationPreset
import openvino.runtime as ov
from torch.utils.data import Dataset, DataLoader
import os

model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
# Tokenize input text
dummy_input = tokenizer("Hello, how are you?", return_tensors="pt")
print("input shape", dummy_input['input_ids'].shape)

# Generate attention mask with correct shape
dummy_attention_mask = torch.ones(dummy_input['input_ids'].shape, dtype=torch.long)

# Create dummy decoder inputs
dummy_decoder_input_ids = torch.ones((1, 1), dtype=torch.long)  # Example dummy input for the decoder
dummy_decoder_attention_mask = torch.ones(dummy_decoder_input_ids.shape, dtype=torch.long)

#Specify output directory and create it if it doesn't exist
output_dir = "blenderbot_openvino_ir"
os.makedirs(output_dir, exist_ok=True)

# Specify the output file path
output_path = os.path.join(output_dir, "blenderbot.onnx")

# Export the model to ONNX format
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_attention_mask, dummy_decoder_input_ids, dummy_decoder_attention_mask),
    output_path,
    opset_version=11,
    input_names=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"],
    output_names=["output"],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'seq_len'},
        'attention_mask': {0: 'batch_size', 1: 'seq_len'},
        'decoder_input_ids': {0: 'batch_size', 1: 'seq_len'},
        'decoder_attention_mask': {0: 'batch_size', 1: 'seq_len'},
        'output': {0: 'batch_size', 1: 'seq_len'}
    }
)
print(f"Model exported to blenderbot_openvino_ir")
