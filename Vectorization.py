from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load Bert model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set model to evaluation mode

# Function to encode texts
def encode(texts):
    # Tokenize the texts in a batch
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
    # Compute the mean of the last hidden state embeddings
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Encode the training and testing data
x_train_encoded = encode(x_train.tolist())  # Convert to list if it's a pandas Series
x_test_encoded = encode(x_test.tolist())

# Check the shapes of the encoded arrays
print(f"Training data encoded shape: {x_train_encoded.shape}")
print(f"Testing data encoded shape: {x_test_encoded.shape}")
