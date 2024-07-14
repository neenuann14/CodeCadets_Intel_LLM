
import torch
from transformers import AutoTokenizer, BlenderbotForConditionalGeneration
from nncf import NNCFConfig
from nncf.quantization import quantize, QuantizationPreset
import openvino.runtime as ov
from torch.utils.data import Dataset, DataLoader
import os
import openvino.runtime as ov
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend

output_dir = "blenderbot_openvino_ir"
# Verify the existence of the model file
model_dir = "blenderbot_openvino_ir"
model_path = os.path.join(model_dir, "blenderbot_ir.xml")
bin_path = os.path.join(model_dir, "blenderbot_ir.bin")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(bin_path):
    raise FileNotFoundError(f"Weights file not found: {bin_path}")

# Load the OpenVINO model
core = ov.Core()
ov_model = core.read_model(model_path, bin_path)
print("openvino model loaded")

# NNCF Configuration
nncf_config = NNCFConfig.from_dict({
    "input_info": {
        "sample_size": [1, 7]  # Adjust to the correct input shape for your model
    },
    "compression": {
        "algorithm": "quantization",
        "initializer": {
            "range": {
                "num_init_samples": 256
            },
            "batchnorm_adaptation": {
                "num_bn_adaptation_samples": 256
            }
        }
    }
})

# Create a dummy calibration dataset
class CalibrationDataset(Dataset):
    def __init__(self, num_samples, batch_size=1):
        self.num_samples = num_samples
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # Generate a tensor with the correct shape [1, 7]
        input_ids = torch.randint(0, 100, (1, 7), dtype=torch.int32)  # Ensure the shape is [1, 7]
        attention_mask = torch.ones((1, 7), dtype=torch.int32)        # Ensure the shape is [1, 7]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_batch_size(self):
        return self.batch_size

    def get_length(self):
        return self.__len__()

    def get_inference_data(self):
        dataloader = DataLoader(self, batch_size=self.batch_size)
        for batch in dataloader:
            input_ids = batch["input_ids"].squeeze(0)
            attention_mask = batch["attention_mask"].squeeze(0)
            yield {"input_ids": input_ids, "attention_mask": attention_mask}

calibration_dataset = CalibrationDataset(num_samples=256, batch_size=1)
print("calibration dataset created")
# Try to explicitly set the backend


backend = get_backend(ov_model)

if backend == BackendType.OPENVINO:
    print("Backend successfully set to OpenVINO")
else:
    raise RuntimeError("Backend is not set to OpenVINO. Current backend: ", backend)

# Quantize the OpenVINO model
print("quantizing model....")
compressed_model = quantize(ov_model, calibration_dataset, preset=QuantizationPreset.MIXED)
print("model quantized.")
# Specify the output paths for the quantized model files
quantized_model_xml_path = os.path.join(output_dir, "blenderbot_quantized.xml")
quantized_model_bin_path = os.path.join(output_dir, "blenderbot_quantized.bin")

# Save the quantized model
ov.serialize(compressed_model, quantized_model_xml_path, quantized_model_bin_path)

print("Quantized OpenVINO model saved as blenderbot_quantized.xml and blenderbot_quantized.bin in the directory:", output_dir)





