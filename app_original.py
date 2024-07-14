#chatbot using original blenderbot 400M distilled model

from flask import Flask, render_template, request, jsonify
import time
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer, pipeline

app = Flask(__name__)

# Load the pretrained model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
chat_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def chat_with_bot(user_input):
    # Check for exit phrases
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        return "Goodbye!", 0.0
    
    start_time = time.time()
    result = chat_pipeline(user_input, max_length=100)
    end_time = time.time()
    response = result[0]['generated_text']
    elapsed_time = end_time - start_time
    return response, elapsed_time

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response, elapsed_time = chat_with_bot(user_input)
    return jsonify({'response': response, 'time': elapsed_time})

if __name__ == '__main__':
    app.run(debug=True)
