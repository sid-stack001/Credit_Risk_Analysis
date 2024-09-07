import torch
import torch.nn as nn
import torch.optim as optim

class CreditRiskNN(nn.Module):
    def __init__(self, input_size):
        super(CreditRiskNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_neural_network(X_train, y_train, X_test, y_test, epochs=100):
    model = CreditRiskNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        test_outputs = model(X_test)
        predictions = (test_outputs > 0.5).float()
    return model, predictions
