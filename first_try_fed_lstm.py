from preprocess import get_all_patient_data
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

patient_data = get_all_patient_data()
time_steps = 10
X_steps = 5  # example value
indiv_models = []
test_data = []

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

for patient in patient_data:
    train_df = patient['train']
    test_df = patient['test']

    # continuous features
    continuous_features = ['cbg', 'basal', 'gsr']
    scaler = MinMaxScaler()

    train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])

    test_df[continuous_features] = scaler.transform(test_df[continuous_features])

    # Reshape features for training data
    X_train, y_train = create_dataset(train_df[['cbg', 'finger', 'basal', 'gsr', 'carbInput', 'bolus']],
                                      train_df['cbg'], time_steps)
    # Reshape features for testing data
    X_test, y_test = create_dataset(test_df[['cbg', 'finger', 'basal', 'gsr', 'carbInput', 'bolus']], test_df['cbg'],
                                    time_steps)


    test_data.append((X_test, y_test))
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=2)

    indiv_models.append(model)
    model.reset_states()

def aggregate_weights_ema(global_model, local_models, alpha=0.5):
    global_weights = global_model.get_weights()
    for model in local_models:
        local_weights = model.get_weights()
        for layer in range(len(global_weights)):
            global_weights[layer] = alpha * global_weights[layer] + (1 - alpha) * local_weights[layer]
    return global_weights

# Aggregate weights
#global_weights = aggregate_weights(indiv_models)


global_model = Sequential()
global_model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
global_model.add(Dense(1))
global_model.compile(loss='mean_squared_error', optimizer='adam')


global_weights = aggregate_weights_ema(global_model, local_models=indiv_models)
global_model.set_weights(global_weights)


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Values')
    plt.plot(predictions, label='Predicted Values', alpha=0.7)
    plt.title(f'Actual vs Predicted CBG')
    plt.xlabel('Time Steps')
    plt.ylabel('CBG Value')
    plt.legend()
    plt.show()

    return mse, mae

total_mse, total_mae, count = 0, 0, 0
for X_test, y_test in test_data:
    mse, mae = evaluate_model(global_model, X_test, y_test)
    total_mse += mse
    total_mae += mae
    count += 1

average_mse = total_mse / count
average_mae = total_mae / count

print(f"Average Mean Squared Error: {average_mse}")
print(f"Average Mean Absolute Error: {average_mae}")

