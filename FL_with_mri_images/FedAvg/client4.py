# import required libraries
import glob
import re
import numpy as np
import flwr as fl
from sklearn.model_selection import train_test_split
import available_datasets
import keras
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf
import random
import util

# Set random seeds
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
tf.compat.v1.set_random_seed(0)
keras.backend.clear_session()

# load the datasets
flavor = 'N4cor_WarpedSegmentationPosteriors2'
delcode = available_datasets.load_DELCODE_data(x_range_from = 32, x_range_to = 161,
                                               y_range_from = 90, y_range_to = 135,
                                               z_range_from = 20, z_range_to = 159, 
                                               flavor = flavor)
delcode_images = delcode['images']
delcode_labels = delcode['labels']
delcode_labels = to_categorical(delcode_labels)

# Split the data into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(delcode_images, delcode_labels, test_size=0.2, random_state=42)
print("Training Input shape\t: {}".format(X_train.shape))
print("Testing Input shape\t: {}".format(X_test.shape))
print("Training Output shape\t: {}".format(Y_train.shape))
print("Testing Output shape\t: {}".format(Y_test.shape))

# Define Flower client
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

        _class_weights = util.get_class_weights(Y_train, Y_train.shape[0])

        # Add EarlyStopping callback
        #early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        for epoch in range(epochs):
            # Train the model for one epoch
            history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size,
                epochs=25,
                verbose=1,
                validation_split=0.1,
                class_weight=_class_weights,
                #callbacks=[early_stopping]
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
        return parameters_prime, num_examples_train, epoch_result

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model with global parameters
        self.model.set_weights(parameters)
        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 8, verbose=0)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

    def get_epoch_results(self):
        return self.epoch_results
   
    def get_final_model(self):
        return self.model

# Load and compile Keras model
model = util.DenseNet(delcode_images.shape[1:], delcode_labels.shape[1])

# Start Flower client
client = DELCODE_Client(model, X_train, Y_train, X_test, Y_test)

fl.client.start_numpy_client(
    server_address="0.0.0.0:9094",
    client=client,
)

# Access and print the results after training
epoch_results = client.get_epoch_results()
print(epoch_results)

final_model = client.get_final_model()
scores = final_model.evaluate(X_test, Y_test)
print("Test %s: %.2f%%" % (final_model.metrics_names[1], scores[1]*100))    

