# === Step 0: Import Libraries ===
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
#%%
# === Step 1: Data Preparation ===
df = pd.read_csv("Final_v2_processed.csv")
test_df = pd.read_csv("Final_test_v2_processed.csv")


#Use LabelEncoder to convert the category variable addr_state to a numeric encoding.
le = LabelEncoder()
df["addr_state"] = le.fit_transform(df["addr_state"])
test_df["addr_state"] = le.transform(test_df["addr_state"])

#Specify the target and feature columns
target_col = "loan_status"
feature_cols = [col for col in df.columns if col != target_col and col != "id"]

# Extracting features and labels (to numpy arrays)
X = df[feature_cols].values
y = df[target_col].values
X_test = test_df[feature_cols].values
y_test = test_df[target_col].values

# Standardised feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Delineate the training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build PyTorch custom dataset classes
class LoanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Construct training, validation and test sets
train_dataset = LoanDataset(X_train, y_train)
val_dataset = LoanDataset(X_val, y_val)
test_dataset = LoanDataset(X_test, y_test)

#%%
# === Step 2: Define MLP Model ===

class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),  # Input to first hidden layer
            nn.ReLU(),
            nn.Linear(64, 32),          # Second hidden layer
            nn.ReLU(),
            nn.Linear(32, 2)            # Output layer (binary classification)
        )

    def forward(self, x):
        return self.model(x)

#%%
# === Step 3:Optuna Hyperparameter Tuning (maximize AUC) ===
def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # DataLoader configuration
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model and optimizer
    model = MLPClassifier(input_size=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model (simplified to 10 epochs)
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation on validation set — focus on probability output + AUC
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            y_true.extend(y_batch.numpy())
            y_prob.extend(probs[:, 1].numpy())  # Probability of predicting "default"

    # Return 1 - AUC (since Optuna minimizes the objective)
    auc = roc_auc_score(y_true, y_prob)
    return 1 - auc

# Run hyperparameter tuning
study = optuna.create_study()
study.optimize(objective, n_trials=20)

# Output best hyperparameters
best_params = study.best_trial.params
print("Best hyperparameters (AUC):", best_params)

#%%
# === Step 4: Train Final Model with Best Parameters ===

# Re-create the data loader (with optimal batch size)
batch_size = best_params['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Re-create the data loader (with optimal batch size)
model = MLPClassifier(input_size=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])

# Save the optimal model state with Early Stopping
best_val_loss = float('inf')
epochs_no_improve = 0
patience = 5
n_epochs = 30  

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    running_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_val_loss += loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Preserve optimal model weights
model.load_state_dict(best_model_state)

# Plotting training/validation loss curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# === Step 5: Threshold Tuning (F1 & G-Mean optimization) on Validation Set ===

# Creating Validation Sets DataLoader
val_loader_eval = DataLoader(val_dataset, batch_size=64)

# Output the predicted probabilities on the validation set using the trained final model
model.eval()
y_true, y_prob = [], []

with torch.no_grad():
    for X_batch, y_batch in val_loader_eval:
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        y_prob.extend(probs[:, 1].numpy())  # Probability of projected ‘default’
        y_true.extend(y_batch.numpy())

y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Try multiple thresholds, recording F1 and G-Mean for each threshold
thresholds = np.arange(0.1, 0.91, 0.01)
f1_scores = []
gmeans = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    gmean = np.sqrt(precision * recall)

    f1_scores.append(f1)
    gmeans.append(gmean)

# Finding the optimal threshold
best_f1_idx = np.argmax(f1_scores)
best_gmean_idx = np.argmax(gmeans)

best_f1_threshold = thresholds[best_f1_idx]
best_gmean_threshold = thresholds[best_gmean_idx]

print(f"Best F1 Threshold (from Validation Set): {best_f1_threshold:.2f}, F1 Score: {f1_scores[best_f1_idx]:.4f}")
print(f"Best G-Mean Threshold (from Validation Set): {best_gmean_threshold:.2f}, G-Mean: {gmeans[best_gmean_idx]:.4f}")

# Visualisation of F1 and G-Mean curves
plt.figure(figsize=(10, 5))
plt.plot(thresholds, f1_scores, label="F1 Score")
plt.plot(thresholds, gmeans, label="G-Mean")
plt.axvline(best_f1_threshold, linestyle='--', color='r', label=f'Best F1 Threshold = {best_f1_threshold:.2f}')
plt.axvline(best_gmean_threshold, linestyle='--', color='g', label=f'Best G-Mean Threshold = {best_gmean_threshold:.2f}')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning on Validation Set - F1 and G-Mean")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# === Step 6: Final Evaluation on All Datasets using optimal thresholds===

# Unified evaluation function
def evaluate_dataset(dataloader, dataset_name, threshold):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            y_prob.extend(probs[:, 1].numpy())
            y_true.extend(y_batch.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculation evaluation metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(y_true, y_prob)

    # Print metrics
    print(f"\n📊 Final Evaluation on {dataset_name} (Threshold = {threshold}):")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"AUC          : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{dataset_name} - Confusion Matrix (Threshold = {threshold})")
    plt.tight_layout()
    plt.show()


# create DataLoader
train_loader_eval = DataLoader(train_dataset, batch_size=64)
val_loader_eval = DataLoader(val_dataset, batch_size=64)
test_loader_eval = DataLoader(test_dataset, batch_size=64)

# Set the final threshold to be used
 # Optional best_f1_threshold or best_gmean_threshold

# Evaluate on all datasets
evaluate_dataset(train_loader_eval, "Training Set", best_f1_threshold)
evaluate_dataset(val_loader_eval, "Validation Set", best_f1_threshold)
evaluate_dataset(test_loader_eval, "Test Set", best_f1_threshold)

evaluate_dataset(train_loader_eval, "Training Set", best_gmean_threshold)
evaluate_dataset(val_loader_eval, "Validation Set", best_gmean_threshold)
evaluate_dataset(test_loader_eval, "Test Set", best_gmean_threshold)

#%%
# === Step 7: Optimal Threshold Comparison - F1 vs G-Mean (Bar Chart Style) ===
# === Step 7: Optimal Threshold Comparison - F1 vs G-Mean (Bar Chart Style) ===
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Build comparison table (based on your test set prediction results)
# Assume you already have performance results for two thresholds:
# results = {
#     'F1 Threshold = 0.18': {...}, # Dictionary containing metrics for F1-optimized threshold
#     'G-Mean Threshold = 0.10': {...} # Dictionary containing metrics for G-Mean-optimized threshold
# }
# The 'results' dictionary should be prepared in a previous step
# Example structure (replace with your actual data):
results = {
    'F1 Threshold = 0.18': {'Accuracy': 0.85, 'Precision': 0.82, 'Recall': 0.90, 'F1 Score': 0.86, 'Specificity': 0.78},
    'G-Mean Threshold = 0.10': {'Accuracy': 0.83, 'Precision': 0.79, 'Recall': 0.92, 'F1 Score': 0.85, 'Specificity': 0.75}
}


# Convert results to DataFrame
df_results = pd.DataFrame(results).T

# Metric names (fixed order)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']

x = np.arange(len(metrics))  # X-axis positions
width = 0.35  # Bar width

# Initialize figure
fig, ax = plt.subplots(figsize=(10, 6))

# Get metric values for both threshold models
# Ensure the keys match exactly what you have in your 'results' dictionary
f1_threshold_key = list(results.keys())[0] # Dynamically get the first key
gmean_threshold_key = list(results.keys())[1] # Dynamically get the second key

f1_values = [df_results.loc[f1_threshold_key, metric] for metric in metrics]
gmean_values = [df_results.loc[gmean_threshold_key, metric] for metric in metrics]

# Plot bars
rects1 = ax.bar(x - width/2, f1_values, width, label=f1_threshold_key, color='dodgerblue')
rects2 = ax.bar(x + width/2, gmean_values, width, label=gmean_threshold_key, color='darkorange')

# Chart settings
ax.set_ylabel('Score')
ax.set_title('Performance Comparison: F1 vs G-Mean Threshold (Test Set)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1) # Set y-axis limit from 0 to 1 for scores
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels function
def add_labels(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', # Format to 2 decimal places
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=9) # Adjust vertical alignment

# Add labels to the bars
add_labels(rects1)
add_labels(rects2)

plt.tight_layout() # Adjust layout to prevent overlapping labels
plt.show()
