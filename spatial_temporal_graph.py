import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from preprocess import get_single_patient_data
from sklearn.preprocessing import MinMaxScaler

# Function to create a graph for a single time point
def create_graph_for_timepoint(data_row):
    G = nx.DiGraph()

    # Adding nodes. Each physiological metric is a node.
    metrics = ['cbg', 'basal', 'bolus', 'carbInput', 'gsr']
    for metric in metrics:
        G.add_node(metric, value=data_row[metric])

    # Adding edges based on assumed physiological relationships
    G.add_edge('carbInput', 'cbg')  # Carbs intake affecting CBG
    G.add_edge('basal', 'cbg')      # Basal insulin affecting CBG
    G.add_edge('bolus', 'cbg')      # Bolus insulin affecting CBG
    G.add_edge('cbg', 'gsr')

    return G

train_patient_data = get_single_patient_data()['train']
test_patient_data = get_single_patient_data()['test']

continuous_features = ['cbg', 'basal', 'gsr']
scaler = MinMaxScaler()

train_patient_data[continuous_features] = scaler.fit_transform(train_patient_data[continuous_features])

test_patient_data[continuous_features] = scaler.transform(test_patient_data[continuous_features])

train_graphs = []
test_graphs = []
for i in range(len(train_patient_data.index)):
    train_graphs.append(create_graph_for_timepoint(train_patient_data.iloc[i]))

for i in range(len(test_patient_data.index)):
    test_graphs.append(create_graph_for_timepoint(test_patient_data.iloc[i]))


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

# number of graphs in each sequence
n = 5

# input-output pairs for training and testing sets
train_inputs, train_outputs = create_input_output_pairs(train_graphs, n)
test_inputs, test_outputs = create_input_output_pairs(test_graphs, n)


###NEURAL NETWORK

import torch.nn as nn
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
        x, edge_index = graph_sequence.x, graph_sequence.edge_index

        # Process with GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index)

        x = geom_nn.global_mean_pool(x, graph_sequence.batch)

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

###TRAINING


from torch_geometric.data import Data

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


import torch
import torch.optim as optim
from torch_geometric.data import DataLoader

# Convert NetworkX graphs to PyTorch Geometric format
train_graphs_pg = convert_networkx_to_pytorch_geometric(train_graphs)

train_dataset = list(zip(train_graphs_pg, train_outputs))

# Use the custom collate function in the DataLoader
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, drop_last=True)

node_feature_size = 1
gcn_hidden_size = 100
lstm_hidden_size = gcn_hidden_size
model = SpatialTemporalGNN(node_feature_size, gcn_hidden_size, lstm_hidden_size)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        graph_sequence, targets = batch
        print('graph_sequence: ', graph_sequence, 'targets: ', targets)
        optimizer.zero_grad()
        predictions = model(graph_sequence)
        #print('q',predictions)
        #print('e',targets.unsqueeze(-1))
        loss = criterion(predictions, targets.unsqueeze(-1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')



#EVALUATION

test_graphs_pg = convert_networkx_to_pytorch_geometric(test_graphs)

model.eval()

# Prepare the DataLoader for test data
test_dataset = list(zip(test_graphs_pg, test_outputs))
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False, drop_last=True)

# Disable gradient calculations
with torch.no_grad():
    total_loss = 0
    for batch in test_loader:
        print(batch)
        graph_sequence, targets = batch
        targets = targets.to('cpu')

        # Forward pass
        predictions = model(graph_sequence)

        # Calculate loss
        loss = criterion(predictions, targets)
        total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)

print(f"Average Test Loss: {avg_loss}")

actual_values = []
predicted_values = []

with torch.no_grad():
    for batch in test_loader:
        graph_sequence, targets = batch
        targets = targets.unsqueeze(-1)
        targets = targets.view(-1, 1)

        # Forward pass
        predictions = model(graph_sequence)

        # Store actual and predicted values
        actual_values.extend(targets.view(-1).tolist())
        predicted_values.extend(predictions.view(-1).tolist())

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label='Actual cbg', color='blue')
plt.plot(predicted_values, label='Predicted cbg', color='red')
plt.title('Actual vs Predicted cbg Values')
plt.xlabel('Time (or Sequence Index)')
plt.ylabel('cbg Value')
plt.legend()
plt.show()