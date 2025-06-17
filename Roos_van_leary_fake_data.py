import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Load dataset
df = pd.read_csv("leary_circle_dataset.csv")

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Generate embeddings
print("Generating BERT embeddings...")
embeddings = [get_bert_embedding(text) for text in tqdm(df["text"])]

X = np.array(embeddings)
y_dominance = df["dominance"].values
y_empathy = df["empathy"].values

# Train-test split
X_train, X_test, y_dom_train, y_dom_test, y_emp_train, y_emp_test = train_test_split(
    X, y_dominance, y_empathy, test_size=0.2, random_state=42
)

# Define neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize model
input_size = X_train.shape[1]
model = NeuralNetwork(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(np.vstack((y_dom_train, y_emp_train)).T, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(np.vstack((y_dom_test, y_emp_test)).T, dtype=torch.float32)

# Train the model
print("Training neural network...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Metrics
print(f"Dominance - R²: {r2_score(y_dom_test, predictions[:, 0]):.4f}, MAE: {mean_absolute_error(y_dom_test, predictions[:, 0]):.4f}")
print(f"Empathy    - R²: {r2_score(y_emp_test, predictions[:, 1]):.4f}, MAE: {mean_absolute_error(y_emp_test, predictions[:, 1]):.4f}")

# Plot
plt.figure(figsize=(10, 10))
plt.scatter(y_emp_test, y_dom_test, color='blue', label='True Values')
plt.scatter(predictions[:, 1], predictions[:, 0], color='red', label='Predicted Values')
plt.axhline(0.5, color='black', linestyle='--')
plt.axvline(0.5, color='black', linestyle='--')
plt.text(0.75, 0.95, 'Dominant-Together', ha='center', fontsize=12)
plt.text(0.25, 0.95, 'Dominant-Against', ha='center', fontsize=12)
plt.text(0.25, 0.05, 'Submissive-Against', ha='center', fontsize=12)
plt.text(0.75, 0.05, 'Submissive-Together', ha='center', fontsize=12)
plt.xlabel('Empathy (0=Against, 1=Together)')
plt.ylabel('Dominance (0=Submissive, 1=Dominant)')
plt.title('Leary Circle - Neural Network Predictions')
plt.legend()
plt.grid(True)
plt.show()
