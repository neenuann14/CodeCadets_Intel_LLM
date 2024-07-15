##BlenderBot with OpenVINO Integration
This project demonstrates the use of OpenVINO to deploy large language models (LLMs) like BlenderBot-400M-distilled across Intel AI laptops. The repository includes scripts to convert, optimize, and quantize the model, as well as to compare the performance of the original and optimized models in a chatbot application.

## Table of Contents
Introduction
Setup
Files and Scripts
Usage
Convert to ONNX
Convert to IR
Quantize Model
Run the Chatbot
Compare Performance
User Interface
License

##Introduction
This project uses the BlenderBot-400M-distilled conversational chatbot model. The goal is to optimize the model for use on Intel AI laptops using OpenVINO.

##Setup
Ensure you have the necessary dependencies installed. You can install the required packages using:

##bash
pip install -r requirements.txt

##Files and Scripts
exporttoonnx.py: Converts the BlenderBot model to an ONNX model and saves it in a new folder blenderbot_openvino_ir.
onnx_to_ir_model.py: Converts the ONNX model to an Intermediate Representation (IR) model.
quantization.py: Quantizes the IR model.
quantized_chatbot.py: Uses the quantized model for the chatbot.
app.py: Provides a user interface for the chatbot using the quantized model, displaying the time taken for each response.
app_original.py: Similar to app.py but uses the original model for comparison.
chatbot.py: Quantizes BlenderBot-9B and uses the quantized model for the chatbot (run in Colab).
static/: Contains HTML, Tailwind CSS, and JavaScript for the user interface.

##Usage
Convert to ONNX
To convert the BlenderBot model to an ONNX model, run:

##bash
python exporttoonnx.py
Convert to IR
To convert the ONNX model to an IR model, run:

bash
python onnx_to_ir_model.py
Quantize Model
To quantize the IR model, run:

bash
python quantization.py
Run the Chatbot
To run the chatbot using the quantized model, run:

bash
python quantized_chatbot.py
Compare Performance
To compare the performance of the original and quantized models, you can use:

bash
python app.py  # For quantized model
python app_original.py  # For original model
These scripts will provide a user interface and display the time taken for each response.

##User Interface
The user interface for the chatbot is created using HTML, Tailwind CSS, and JavaScript. The static files are located in the static/ directory.

##License
This project is licensed under the MIT License. See the LICENSE file for details.
