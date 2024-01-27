import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import get_all_patient_data
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

patient_data = get_all_patient_data()
time_steps = 10
X_steps = 5
indiv_models = []
test_data = []

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)


class CustomDataset(Dataset):

    def __init__(self, inputs, targets):
        assert isinstance(inputs, np.ndarray), "Inputs should be a NumPy array"
        assert isinstance(targets, np.ndarray), "Targets should be a NumPy array"
        self.inputs = torch.from_numpy(inputs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def LoadData():
    all_loaders = []
    for patient in patient_data:
        train_df = patient['train']

        # continuous features
        continuous_features = ['cbg', 'basal', 'gsr']
        scaler = MinMaxScaler()

        train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])

        # Reshape features for training data
        X_train, y_train = create_dataset(train_df[['cbg', 'finger', 'basal', 'gsr', 'carbInput', 'bolus']],
                                          train_df['cbg'], time_steps)

        tdata = CustomDataset(X_train, y_train)
        train_loader = DataLoader(tdata, batch_size=12, shuffle=True, drop_last=True)
        all_loaders.append(train_loader)

    return all_loaders

all_loaders = LoadData()

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0, c0 = self.init_hidden(x.size(0))
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, batch_size):
        h0 = torch.zeros(1, batch_size, self.hidden_dim)
        c0 = torch.zeros(1, batch_size, self.hidden_dim)
        return [t for t in (h0, c0)]

global_model = LSTMModel(input_dim=6, hidden_dim=12)


import numpy as np

def select_clients(num_clients, fraction):
    selected_clients_indices = np.random.choice(range(num_clients), int(num_clients * fraction), replace=False)
    return selected_clients_indices


def local_update(client_model, optimizer, train_loader, epochs=1):
    client_model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.cpu(), target.cpu()
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.MSELoss()(output, target.view(-1, 1))
            loss.backward()
            optimizer.step()
    return client_model.state_dict(), loss.item()

def aggregate_global_model(global_model, client_models, client_weights):
    global_state_dict = global_model.state_dict()
    for k in global_state_dict.keys():
        global_state_dict[k] = torch.stack([client_weights[i] * client_models[i][k] for i in range(len(client_models))], 0).sum(0)
    global_model.load_state_dict(global_state_dict)
    return global_model


num_clients = 12
num_selected = 3
num_rounds = 20
client_fraction = num_selected / num_clients


for round in range(num_rounds):
    selected_clients = select_clients(num_clients, client_fraction)

    client_models = []
    client_losses = []
    for client in selected_clients:

        client_train_loader = all_loaders[client]

        # Copy global model to client model
        client_model = LSTMModel(input_dim=6, hidden_dim=12)
        client_model.load_state_dict(global_model.state_dict())

        # Optimizer for client model
        optimizer = optim.Adam(client_model.parameters(), lr=0.01)

        # local training
        client_state_dict, client_loss = local_update(client_model, optimizer, client_train_loader, epochs=5)

        print(client_loss)

        client_models.append(client_state_dict)
        client_losses.append(client_loss)
    print('1d')

    # Aggregate updates
    global_model = aggregate_global_model(global_model, client_models, [1 / num_selected] * num_selected)




from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cpu(), target.cpu()
            output = model(data)
            predictions.extend(output.view(-1).cpu().numpy())
            actuals.extend(target.view(-1).cpu().numpy())

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    return mse, mae


client_mse = []
client_mae = []

def LoadData_test():
    all_loaders = []
    for patient in patient_data:
        train_df = patient['test']

        # continuous features
        continuous_features = ['cbg', 'basal', 'gsr']
        scaler = MinMaxScaler()

        train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])

        # Reshape features for training data
        X_train, y_train = create_dataset(train_df[['cbg', 'finger', 'basal', 'gsr', 'carbInput', 'bolus']],
                                          train_df['cbg'], time_steps)

        tdata = CustomDataset(X_train, y_train)
        train_loader = DataLoader(tdata, batch_size=12, shuffle=True, drop_last=True)
        all_loaders.append(train_loader)

    return all_loaders

test_data_loaders = LoadData_test()

for client_id, test_loader in enumerate(test_data_loaders):
    mse, mae = evaluate_model(global_model, test_loader)
    client_mse.append(mse)
    client_mae.append(mae)

# Aggregate metrics
average_mse = sum(client_mse) / len(client_mse)
average_mae = sum(client_mae) / len(client_mae)

print(f"Average MSE across all clients: {average_mse}")
print(f"Average MAE across all clients: {average_mae}")


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Assuming client_mse and client_mae contain MSE and MAE for each client respectively
client_ids = [f'Client {i}' for i in range(len(client_mse))]

actual_values = []
predicted_values = []

selected_patient_predictions, selected_patient_actuals = [], []

global_model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for data, target in test_data_loaders[0]:
        data = data.cpu()
        output = global_model(data)
        selected_patient_predictions.extend(output.view(-1).cpu().numpy())
        selected_patient_actuals.extend(target.view(-1).cpu().numpy())

sampling_rate = 10

# Sampled data points
sampled_indices = range(0, len(selected_patient_actuals), sampling_rate)
sampled_actuals = [selected_patient_actuals[i] for i in sampled_indices]
sampled_predictions = [selected_patient_predictions[i] for i in sampled_indices]


plt.figure(figsize=(10, 6))
plt.plot(sampled_indices, sampled_actuals, label='Actual CBG')
plt.plot(sampled_indices, sampled_predictions, label='Predicted CBG')
plt.title('Actual vs Predicted CBG Values (Sampled)')
plt.xlabel('Time/Sequence Index')
plt.ylabel('CBG Value')
plt.legend()
plt.show()

