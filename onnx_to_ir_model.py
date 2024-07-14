
import torch
import os
import openvino.runtime as ov


output_dir = "blenderbot_openvino_ir"
# Load the ONNX model
core = ov.Core()
onnx_model_path = "C:/Users/Lenovo\Desktop/INTEL_CHATBOT/blenderbot_openvino_ir/blenderbot.onnx"
onnx_model = core.read_model(onnx_model_path)

# Specify the output paths for the OpenVINO IR format files
ir_xml_path = os.path.join(output_dir, "blenderbot_ir.xml")
ir_bin_path = os.path.join(output_dir, "blenderbot_ir.bin")

# Convert the ONNX model to OpenVINO IR format
# Note: No need to compile the model for conversion
ov.serialize(onnx_model, ir_xml_path, ir_bin_path)

print("Model has been successfully converted to OpenVINO IR format and saved to", output_dir)


