import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from argparse import Namespace
from itertools import product
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import set_seed
from models.TabLLM import TabLLM
from utils import load_dataset
from imblearn.over_sampling import SMOTE

# early stopping
class EarlyStopping:
    def __init__(self, patience=3, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# output path
DATASET = "parkinsons"
RESULTS_DIR = f"results/{DATASET}"
os.makedirs(RESULTS_DIR, exist_ok=True)
set_seed(42)

# load the data
train_x, val_x, test_x, train_y, val_y, test_y = load_dataset(DATASET)


# grid search
dropouts = [0.2, 0.3]
lrs = [1e-3, 2e-4]
llm_layers_list = [1, 2]

# best
best_val_acc = 0.0
best_model_state = None
best_hparams = {}
best_train_losses = []
best_val_losses = []
best_val_accuracies = []

# loop for grid search
for dropout, lr, llm_layers in product(dropouts, lrs, llm_layers_list):
    print(f"\nTrying: dropout={dropout}, lr={lr}, llm_layers={llm_layers}")

    config = Namespace(
        num_classes=len(torch.unique(train_y)),
        n_vars=train_x.shape[1],
        dropout=dropout,
        task_name="classification",
        llm_layers=llm_layers,
    )
    model = TabLLM(config, prompt=f"Classify {DATASET} dataset row:")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=3)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = (val_preds == val_y).float().mean().item()
            val_losses.append(val_loss.item())
            val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss.item():.4f}")

        early_stopper(val_loss.item())
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    if val_accuracies[-1] > best_val_acc:
        best_val_acc = val_accuracies[-1]
        best_model_state = copy.deepcopy(model.state_dict())
        best_hparams = {"dropout": dropout, "lr": lr, "llm_layers": llm_layers}
        best_train_losses = train_losses
        best_val_losses = val_losses
        best_val_accuracies = val_accuracies

# saving best model 
torch.save(best_model_state, f"{RESULTS_DIR}/{DATASET}_best_model.pt")
with open(f"{RESULTS_DIR}/{DATASET}_best_model_params.txt", "w") as f:
    f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n")
    f.write("Best Hyperparameters:\n")
    for k, v in best_hparams.items():
        f.write(f"{k}: {v}\n")

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
print("Saved best model and hyperparameters.")

# on test set
print("\nEvaluating best model on test set...")
# === Rebuild model with best hyperparameters before loading ===
best_config = Namespace(
    num_classes=len(torch.unique(train_y)),
    n_vars=train_x.shape[1],
    dropout=best_hparams["dropout"],
    task_name="classification",
    llm_layers=best_hparams["llm_layers"],
)
model = TabLLM(best_config, prompt=f"Classify {DATASET} dataset row:")
model.load_state_dict(torch.load(f"{RESULTS_DIR}/{DATASET}_best_model.pt"))
model.eval()
with torch.no_grad():
    test_outputs = model(test_x)
    test_preds = torch.argmax(test_outputs, dim=1)
    test_acc = (test_preds == test_y).float().mean().item()
    print(f"Test Accuracy on {DATASET}: {test_acc:.4f}")

    cm = confusion_matrix(test_y.cpu(), test_preds.cpu())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(f"{RESULTS_DIR}/{DATASET}_confusion_matrix.png")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# === Plot loss and accuracy curves ===
if len(best_train_losses) > 1:
    epochs = list(range(1, len(best_train_losses) + 1))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, best_train_losses, label="Train Loss")
    plt.plot(epochs, best_val_losses, label="Val Loss")
    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, best_val_accuracies, label="Val Accuracy", color="green")
    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{DATASET}_training_metrics.png")
    plt.show(block=False)
    plt.pause(2)
    plt.close()
else:
    print("Not enough epochs to generate plots.")
