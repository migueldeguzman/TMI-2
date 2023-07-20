import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model_and_tokenizer(model_path):
    # Initialize the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return tokenizer, model

def string_to_tensor(activation_values):
    # Replace spaces with commas and split the string into a list of floats
    activation_list = [float(val) for val in activation_values.replace(' ', ',').split(',')]
    
    # Convert the list into a tensor
    tensor = torch.tensor([activation_list])
    return tensor

def get_top_tokens(activation_values, tokenizer, model, temperature=.50, k=20):
    # Convert the string of activation values into a tensor
    tensor = string_to_tensor(activation_values)
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(tensor/temperature, dim=-1)
    
    # Get the top k probabilities and their corresponding indices
    top_k_probs, top_k_indices = torch.topk(probabilities, k)
    
    # Convert the indices into tokens
    top_k_tokens = [tokenizer.decode([index.item()]) for index in top_k_indices[0]]
    
    return top_k_tokens, top_k_probs[0].tolist()

# Initialize the model and tokenizer
model_path = 'gpt2-xl' # or path to your fine-tuned model
tokenizer, model = load_model_and_tokenizer(model_path)

# Predefined activation values for demonstration
activation_values = "add activation values here"

# Get and print the top tokens and their probabilities
top_k_tokens, top_k_probs = get_top_tokens(activation_values, tokenizer, model)
for i in range(len(top_k_tokens)):
    print(f"Token: {top_k_tokens[i]}, Probability: {top_k_probs[i]}")
