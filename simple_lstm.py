from preprocess import get_single_patient_data
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train_df = get_single_patient_data()['train']
test_df = get_single_patient_data()['test']

# Selecting continuous features
continuous_features = ['cbg', 'basal', 'gsr']
scaler = MinMaxScaler()

train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])

test_df[continuous_features] = scaler.transform(test_df[continuous_features])

#window sliding
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10

# Reshape features for training data
X_train, y_train = create_dataset(train_df[['cbg', 'finger', 'basal', 'gsr', 'carbInput', 'bolus']], train_df['cbg'], time_steps)
# Reshape features for testing data
X_test, y_test = create_dataset(test_df[['cbg', 'finger', 'basal', 'gsr', 'carbInput', 'bolus']], test_df['cbg'], time_steps)


X_steps = 5

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=2)

# Prepare test data
test_features = test_df[['cbg', 'finger', 'basal', 'gsr', 'carbInput', 'bolus']].values
test_target = np.roll(test_df['cbg'].values, -X_steps)
test_features = test_features[:-X_steps, :]
test_target = test_target[X_steps:]
test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))

# Evaluate model
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

num_points_to_plot = 2000

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_target[:num_points_to_plot], label='Actual')
plt.plot(predictions[:num_points_to_plot], label='Predicted', alpha=0.7)
plt.title('CBG Prediction')
plt.xlabel('Time Steps')
plt.ylabel('CBG Value')
plt.legend()
plt.show()
