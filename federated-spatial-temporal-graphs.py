import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import get_all_patient_data
from torch_geometric.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from torch_geometric.data import Data


# Function to create a graph for a single time point
def create_graph_for_timepoint(data_row):
    G = nx.DiGraph()

    # Adding nodes. Each physiological metric is a node.
    metrics = ['cbg', 'basal', 'bolus', 'carbInput', 'gsr']
    for metric in metrics:
        G.add_node(metric, value=data_row[metric])

    # Adding edges based on assumed physiological relationships
    # if data_row['carbInput'] != -1:
    G.add_edge('carbInput', 'cbg')  # Carbs intake affecting CBG
    G.add_edge('basal', 'cbg')      # Basal insulin affecting CBG
    G.add_edge('bolus', 'cbg')      # Bolus insulin affecting CBG
    G.add_edge('cbg', 'gsr')

    return G

def create_input_output_pairs(graphs, n, steps_ahead=5):
    """
    Create input-output pairs for training the GNN.
    Each input is a sequence of n graphs, and the output is the 'cbg' value steps_ahead time steps later.
    """
    inputs, outputs = [], []
    for i in range(len(graphs) - n - steps_ahead + 1):
        input_graphs = graphs[i:i + n]
        output_value = graphs[i + n + steps_ahead - 1].nodes['cbg']['value']
        inputs.append(input_graphs)
        outputs.append(output_value)
    return inputs, outputs

def convert_networkx_to_pytorch_geometric(nx_graphs):
    """
    Converts a list of NetworkX graphs into PyTorch Geometric Data objects.
    """
    pyg_graphs = []

    for nx_graph in nx_graphs:
        # Get node features
        node_features = []
        for _, node_data in nx_graph.nodes(data=True):
            node_features.append([node_data['value']])
        node_features = torch.tensor(node_features, dtype=torch.float)

        # Get edge indices
        edge_index = []
        for source, target in nx_graph.edges():
            edge_index.append([source, target])

        # dictionary that maps integer to its string value
        unique_strings = set(item for sublist in edge_index for item in sublist)
        # Map each unique string to an integer
        string_to_int_map = {string: i for i, string in enumerate(unique_strings)}
        transformed_list = [[string_to_int_map[string] for string in sublist] for sublist in edge_index]
        edge_index = torch.tensor(transformed_list, dtype=torch.long).t().contiguous()

        # Create PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index)
        pyg_graphs.append(data)

    return pyg_graphs


patient_data = get_all_patient_data()
time_steps = 10
X_steps = 5
indiv_models = []
test_data = []



def LoadData():
    all_loaders = []
    for patient in patient_data:
        train_graphs = []
        train_df = patient['train']

        # Selecting continuous features
        continuous_features = ['cbg', 'basal', 'gsr']
        scaler = MinMaxScaler()

        train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])

        for i in range(len(train_df.index)):
            train_graphs.append(create_graph_for_timepoint(train_df.iloc[i]))

        # Reshape features for training data
        X_train, y_train = create_input_output_pairs(train_graphs, n=5)

        train_graphs_pg = convert_networkx_to_pytorch_geometric(train_graphs)

        # Create a DataLoader
        train_dataset = list(zip(train_graphs_pg, y_train))

        # Use the custom collate function in the DataLoader
        train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, drop_last=True)
        all_loaders.append(train_loader)

    return all_loaders

all_loaders = LoadData()

import torch_geometric.nn as geom_nn

class SpatialTemporalGNN(nn.Module):
    def __init__(self, node_feature_size, gcn_hidden_size, lstm_hidden_size, num_gcn_layers=2):
        super(SpatialTemporalGNN, self).__init__()

        # GCN Layers
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            gcn_layer = geom_nn.GCNConv(node_feature_size, gcn_hidden_size)
            self.gcn_layers.append(gcn_layer)
            node_feature_size = gcn_hidden_size  # Update feature size for the next layer

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=gcn_hidden_size, hidden_size=lstm_hidden_size, batch_first=True)

        # Output Layer
        self.output_layer = nn.Linear(lstm_hidden_size, 1)  # Predicting a single value

    def forward(self, graph_sequence):
        # graph_sequence is a batched Data object
        x, edge_index = graph_sequence.x, graph_sequence.edge_index  # Node features and edge indices for the entire batch

        # Process with GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)

        # Here's an example using mean pooling to aggregate node features for each graph in the batch
        x = geom_nn.global_mean_pool(x, graph_sequence.batch)  # Shape: [num_graphs, gcn_hidden_size]

        sequence_length = 5  # As defined
        batch_size = 30

        x = x.view(batch_size, sequence_length, -1)

        # LSTM layer for temporal processing
        lstm_out, _ = self.lstm(x)

        # Final prediction
        prediction = self.output_layer(lstm_out[:, -1, :])
        return prediction

        # Final prediction
        prediction = self.output_layer(lstm_out[:, -1, :])
        return prediction


node_feature_size = 1
gcn_hidden_size = 100
lstm_hidden_size = gcn_hidden_size
global_model = SpatialTemporalGNN(node_feature_size, gcn_hidden_size, lstm_hidden_size)


import random

def select_random_sublist(num_clients, num_selected):
    if num_clients % num_selected != 0:
        raise ValueError("num_clients must be divisible by num_selected")
    clients = list(range(num_clients))
    sublists = [clients[i:i + num_selected] for i in range(0, num_clients, num_selected)]

    return random.choice(sublists)


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


num_patients = 12 #number of patients
num_selected = 4 #number of patients per client
num_rounds = 5


for round in range(num_rounds):
    selected_clients = select_random_sublist(num_patients, num_selected)

    client_models = []
    client_losses = []
    for client in selected_clients:
        client_train_loader = all_loaders[client]

        # Copy global model to client model
        client_model = SpatialTemporalGNN(node_feature_size, gcn_hidden_size, lstm_hidden_size)
        client_model.load_state_dict(global_model.state_dict())

        # Optimizer for client model
        optimizer = optim.Adam(client_model.parameters(), lr=0.01)

        client_state_dict, client_loss = local_update(client_model, optimizer, client_train_loader, epochs=5)

        print(client_loss)

        client_models.append(client_state_dict)
        client_losses.append(client_loss)
    print('round')

    # Aggregate updates
    global_model = aggregate_global_model(global_model, client_models, [1 / num_selected] * num_selected) #weights / aggregation function could be improved




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

def LoadDatatest():
    all_loaders = []
    for patient in patient_data:
        train_graphs = []
        train_df = patient['test']

        # continuous features
        continuous_features = ['cbg', 'basal', 'gsr']
        scaler = MinMaxScaler()

        train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])

        for i in range(len(train_df.index)):
            train_graphs.append(create_graph_for_timepoint(train_df.iloc[i]))

        # Reshape features for training data
        X_train, y_train = create_input_output_pairs(train_graphs, n=5)

        train_graphs_pg = convert_networkx_to_pytorch_geometric(train_graphs)

        train_dataset = list(zip(train_graphs_pg, y_train))

        train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, drop_last=True)
        all_loaders.append(train_loader)

    return all_loaders

test_data_loaders = LoadDatatest()

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

client_ids = [f'Client {i}' for i in range(len(client_mse))]


actual_values = []
predicted_values = []

selected_patient_predictions, selected_patient_actuals = [], []

global_model.eval()

with torch.no_grad():
    for data, target in test_data_loaders[5]:
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
