import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from proposed_optimizer import AGSGD, OurProposedSGD


# ---- Step 1: Set up Data ---- #
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---- Step 2: Define a Simple Model ---- #
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

# ---- Step 3: Define Optimizers to Compare ---- #
optimizers_dict = {
    "Vanilla": lambda model: optim.SGD(model.parameters(), lr=0.01),
    "AdaGrad": lambda model: optim.Adagrad(model.parameters(), lr=0.01),
    "AdaDelta": lambda model: optim.Adadelta(model.parameters(), lr=1.0),
    "AG-SGD (Paper Proposed)": lambda model: AGSGD(model.parameters(), lr=0.001, s=1.2, d=0.95, iter_decay=10),
    "Our Proposed": lambda model: OurProposedSGD(model.parameters(), lr=0.001, s=1.2, d=0.95, iter_decay=10)
}

# ---- Step 4: Training + Testing Loop ---- #
results = {}  # To hold 50 test costs per optimizer
criterion = nn.CrossEntropyLoss()

for opt_name, opt_fn in optimizers_dict.items():
    print(f"\nTraining with {opt_name}...")
    model = SimpleNN()
    optimizer = opt_fn(model)
    test_costs = []

    for epoch in range(50):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                output = model(x_test)
                loss = criterion(output, y_test)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        test_costs.append(avg_test_loss)
        print(f"Epoch {epoch+1}: Test Cost = {avg_test_loss:.4f}")

    results[opt_name] = test_costs

# ---- Step 5: Save Results ---- #
df = pd.DataFrame(results)
df.index.name = "Iteration"
df.to_csv("test_costs_per_optimizer.csv")
print("\nSaved test costs to 'test_costs_per_optimizer.csv'.")

# ---- Step 6: Extract 5 Minimal Costs and Save Stats ---- #
stats = []
for name, costs in results.items():
    top5 = sorted(costs)[:5]
    avg = np.mean(top5)
    std = np.std(top5)
    var = np.var(top5)
    stats.append([name] + top5 + [avg, std, var])

columns = ["Optimizer", "1st", "2nd", "3rd", "4th", "5th", "Average", "StdDev", "Variance"]
stats_df = pd.DataFrame(stats, columns=columns)
stats_df.to_csv("minimal_cost_stats.csv", index=False)
print("\nSaved 5 minimal cost stats to 'minimal_cost_stats.csv'.")

# ---- Step 7: Plot Graphs ---- #
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
for col in df.columns:
    plt.plot(df.index, np.array(df[col]) * 10000, label=col, marker='o', markersize=2)
plt.xlabel("Epoch")
plt.ylabel("Cost (testing) × 10,000")
plt.title("Test Cost Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_costs_over_epochs.png")
plt.show()

# Boxplot for 5 minimal costs
box_data = []
box_labels = []
for i, row in stats_df.iterrows():
    for val in row[1:6]:
        box_data.append(val * 10000)
        box_labels.append(row["Optimizer"])

plt.figure(figsize=(12, 6))
sns.boxplot(x=box_labels, y=box_data)
plt.xlabel("Optimizer")
plt.ylabel("Cost (testing) × 10,000")
plt.title("Distribution of the 5 Minimal Costs")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_minimal_costs.png")
plt.show()
