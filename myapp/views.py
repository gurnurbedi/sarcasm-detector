# views.py
import os
import re
import torch
import torch.nn as nn
from django.conf import settings
from django.shortcuts import render

# Define your LSTMModel class
class LSTNModel(nn.Module):
    def __init__(self):
        super(LSTNModel, self).__init__()
        # Define your layers here
        # Example: self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, x):
        # Define the forward pass
        return x  # Replace with actual implementation

class SimpleTokenizer:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        # Add special tokens
        self.word_to_idx['<pad>'] = 0
        self.word_to_idx['<unk>'] = 1
        self.idx_to_word[0] = '<pad>'
        self.idx_to_word[1] = '<unk>'
        
    def tokenize(self, text):
        # Simple tokenization: lowercase and split on whitespace/punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def encode(self, tokens, max_length=128):
        # Convert tokens to indices
        indices = [self.word_to_idx.get(token, self.word_to_idx['<unk>']) for token in tokens]
        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([self.word_to_idx['<pad>']] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
        return indices

def load_model_and_tokenizer():
    try:
        model_path = os.path.join(settings.BASE_DIR, 'myapp', 'llm', 'LLMSarc_LSTN_Dataset3')  # Ensure the correct file extension
        print(f"Attempting to load model from: {model_path}")
        print(f"File exists: {os.path.exists(model_path)}")
        
        # Create an instance of the model
        model = LSTNModel()
        
        # Load the model state
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Initialize tokenizer
        tokenizer = SimpleTokenizer()
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def predict_sarcasm(text, model, tokenizer):
    try:
        # Tokenize and encode the text
        tokens = tokenizer.tokenize(text)
        encoded = tokenizer.encode(tokens)
        
        # Convert to tensor
        input_tensor = torch.tensor([encoded])  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            prediction = torch.argmax(outputs, dim=1)
        
        return "Sarcastic" if prediction.item() == 1 else "Not Sarcastic"
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

def home(request):
    return render(request, 'home.html')

def intro(request):
    return render(request, 'intro.html')

def page2(request):
    if request.method == 'POST':
        try:
            user_text = request.POST.get('user_text', '')
            
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer()
            
            # Get prediction
            prediction_result = predict_sarcasm(user_text, model, tokenizer)
            
            return render(request, 'page2.html', {
                'user_text': user_text,
                'prediction_result': prediction_result
            })
            
        except Exception as e:
            error_message = f"Error processing text: {str(e)}"
            print(error_message)  # For server logs
            return render(request, 'page2.html', {
                'user_text': user_text if 'user_text' in locals() else '',
                'prediction_result': error_message
            })
    
    return render(request, 'page2.html')