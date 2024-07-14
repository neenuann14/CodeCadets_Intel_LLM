import openvino.runtime as ov
from transformers import BlenderbotTokenizer
import os
import numpy as np

# Load the tokenizer
model_dir = "blenderbot_openvino_ir"
tokenizer = BlenderbotTokenizer.from_pretrained(model_dir)

# Load the OpenVINO model
core = ov.Core()
model_path = os.path.join(model_dir, "blenderbot_quantized.xml")
compiled_model = core.compile_model(model=model_path, device_name="CPU")
infer_request = compiled_model.create_infer_request()

def chat_with_bot(user_input):
    # Check for exit phrases
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        return "Goodbye!", 0.0

    # Tokenize input
    inputs = tokenizer(user_input, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Ensure input size matches the model's expected input size
    max_length = compiled_model.input(0).shape[1]
    padded_input_ids = np.pad(input_ids, ((0, 0), (0, max_length - input_ids.shape[1])), constant_values=tokenizer.pad_token_id)
    padded_attention_mask = np.pad(attention_mask, ((0, 0), (0, max_length - attention_mask.shape[1])), constant_values=0)

    # Run inference
    output = infer_request.infer(inputs={"input_ids": padded_input_ids, "attention_mask": padded_attention_mask})

    # Decode output
    output_ids = np.argmax(output[list(output)[0]], axis=-1)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response

def chat():
    print("Welcome to the Chatbot. Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit","bye","goodbye"]:
            print("Chatbot: Goodbye!")
            break
        response = chat_with_bot(user_input)
        print(f"Chatbot: {response}")

chat()
