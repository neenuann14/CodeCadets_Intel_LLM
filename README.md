This project is to study the use of openvino in using large LLM models so that it can be used across all intel AI laptops .
We have used [blenderbot-400M-distilled ](https://huggingface.co/facebook/blenderbot-400M-distill) model as our LLM model. It is a conversational chatbot.
In this repository , exporttoonnx.py converts model to onnx model and saves it in a new folder blenderbot_openvino_ir .
This onnx model is converted to ir model using onnx_to_ir_model.py.
The ir model is quantized using quantization.py.
The quantized model is used for chatbot in quantized_chatbot.py.
app.py gives user interface for chatbot using quantised model at the same time giving time taken for each response.
app_original.py gives similar user interface but uses original model so that we can compare the time taken for each response.

