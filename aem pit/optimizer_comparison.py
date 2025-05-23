import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from proposed_optimizer import AngleSGD


# Set random seeds
torch.manual_seed(0)
np.random.seed(0)

def F(a, s=1):
    k = 15.0  # sharper than C2 but smoother than C3
    a_tensor = torch.tensor(a, dtype=torch.float32)
    return -s * (2 / (1 + torch.exp(-k * (2 * a_tensor - 1))) - 1)

x = np.linspace(0, 1, 100)
y = F(x, s=1)

x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define model
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Optimizers to compare
def get_optimizers(model):
    return {
        "Vanilla": optim.SGD(model.parameters(), lr=0.01),
        "Momentum": optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        "RMSprop": optim.RMSprop(model.parameters(), lr=0.01),
        "Adam": optim.Adam(model.parameters(), lr=0.01),
        "Nadam": optim.NAdam(model.parameters(), lr=0.01),
        "AdaMax": optim.Adamax(model.parameters(), lr=0.01),
        "AdaDelta": optim.Adadelta(model.parameters(), lr=1.0),
        "AdaGrad": optim.Adagrad(model.parameters(), lr=0.01),
        "AMSGrad": optim.Adam(model.parameters(), lr=0.01, amsgrad=True),
        "Proposed": AngleSGD(model.parameters(), lr=0.0005, s=1.0, d=0.5, iter_decay=5)  # Replace with your method
    }

num_epochs = 50
iterations_per_epoch = 100  # Number of iterations per epoch
total_iterations = num_epochs * iterations_per_epoch
test_costs = {opt_name: [] for opt_name in get_optimizers(SimpleLinearModel()).keys()}
min_costs = []

for opt_name in test_costs.keys():
    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = get_optimizers(model)[opt_name]
    iteration_costs = []

    for epoch in range(num_epochs):
        for iteration in range(iterations_per_epoch):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                test_output = model(x_tensor)
                test_loss = criterion(test_output, y_tensor).item()
                iteration_costs.append(test_loss)

    test_costs[opt_name] = iteration_costs
    min_costs.append([opt_name] + sorted(iteration_costs)[-5:][::-1])

# Save cost per iteration
cost_df = pd.DataFrame(test_costs)
cost_df.to_csv("test_costs_per_optimizer.csv")

# Compute statistics
stats = []
for row in min_costs:
    opt_name = row[0]
    values = row[1:]
    avg = np.mean(values)
    std = np.std(values)
    var = np.var(values)
    stats.append([opt_name] + values + [avg, std, var])

columns = ["Optimizer", "Min1", "Min2", "Min3", "Min4", "Min5", "Average", "Std Dev", "Variance"]
stats_df = pd.DataFrame(stats, columns=columns)
stats_df.to_csv("minimal_cost_stats.csv", index=False)

# Plot cost vs iteration
plt.figure(figsize=(12, 6))
iterations = range(total_iterations)
for col in cost_df.columns:
    plt.plot(iterations, cost_df[col] * 10000, label=col, marker='o', markersize=2)
plt.xlabel("Iteration")
plt.ylabel("Cost(testing) scaled by 10,000")
plt.legend()
plt.title("Iteration-wise Cost Comparison")
plt.grid(True)
plt.tight_layout()
plt.savefig("cost_vs_iteration.png")
plt.show()

# Boxplot
box_data = []
box_labels = []
for i, row in stats_df.iterrows():
    for val in row[1:6]:
        box_data.append(val * 10000)
        box_labels.append(row["Optimizer"])

plt.figure(figsize=(12, 6))
sns.boxplot(x=box_labels, y=box_data)
plt.xlabel("Optimizer")
plt.ylabel("Cost(testing) scaled by 10,000")
plt.title("Distributions of the 5 Minimal Costs")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_minimal_costs.png")
plt.show()

# Save LaTeX table
latex_table = stats_df.to_latex(index=False, float_format="%.4f", caption="Statistical Information of the 5 Minimal Costs", label="tab:min_costs")
with open("minimal_costs_table.tex", "w") as f:
    f.write(latex_table)
print("All files saved successfully.")
