# import required libraries
import flwr as fl
from model import create_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import numpy as np
import keras
from keras.callbacks import EarlyStopping
import random

# Set random seeds
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
tf.compat.v1.set_random_seed(0)
keras.backend.clear_session()


# Load the CSV data into a pandas dataframe
data = pd.read_csv('FastSurfer_Volumes_combined_normalized_to_TIV.csv')

# Discard unnecessary rows
data = data[~data['sample'].isin(['ADNI3','AIBL', 'EDSD', 'ADNI2/GO'])]
data = data[data.grp != 'bvFTD']
data['grp']= data['grp'].replace('CN', 0).replace('AD', 1).replace('MCI', 1)

# Discard unnecessary columns
data = data.drop(['sid', 'sex1f', 'age', 'education_y', 'MMSE', 'MRI_field_strength',], axis=1)

# Divide float64 columns by 'eTIV'
float_columns = data.select_dtypes(include=['float64']).columns
data[float_columns] = data[float_columns].div(data['eTIV'], axis=0)
data = data.drop(['eTIV'], axis=1)

# Separate target variable and input features
y = data['grp']
X = data.drop(['sample', 'grp'], axis=1)

# Split the data into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scalining the data(normalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
    
class DELCODE_Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.epoch_results = []        

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        class_weights =compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weight_dict=dict(enumerate(class_weights))

        # Add EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        for epoch in range(epochs):
            # Train the model for one epoch
            history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size,
                epochs=80,
                verbose=1,
                validation_split=0.1,
                class_weight=class_weight_dict,
                callbacks=[early_stopping],
            )
            # Store the results for this epoch
            epoch_result = {
                "loss": history.history["loss"][0],
                "accuracy": history.history["accuracy"][0],
                "val_loss": history.history["val_loss"][0],
                "val_accuracy": history.history["val_accuracy"][0],
            }
            # Append results to the epoch_results list
            self.epoch_results.append(epoch_result)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        return parameters_prime, num_examples_train, epoch_result  # Return last epoch result

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 8)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


    def get_epoch_results(self):
        return self.epoch_results
    
    def get_final_model(self):
        return self.model

# Load and compile Keras model
model = create_model()

# Start Flower client
client = DELCODE_Client(model, X_train, Y_train, X_test, Y_test)
fl.client.start_numpy_client(
    server_address="0.0.0.0:9098",
    client=client,
)

# Access and print the results after training
epoch_results = client.get_epoch_results()

final_model = client.get_final_model()
scores = final_model.evaluate(X_test, Y_test) 
print("Test %s: %.2f%%" % (final_model.metrics_names[1], scores[1]*100))
for epoch, results in enumerate(epoch_results, start=1):
    print(f"Epoch {epoch} Results:")
    print(f"Loss: {results['loss']}")
    print(f"Accuracy: {results['accuracy']}")
    print(f"Validation Loss: {results['val_loss']}")
    print(f"Validation Accuracy: {results['val_accuracy']}")
