# import required libraries
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import create_model
import keras
import tensorflow as tf
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
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
data = data[~data['sample'].isin(['ADNI2/GO','AIBL', 'EDSD', 'DELCODE'])]
data = data[data.grp != 'bvFTD']

# Discard unnecessary columns
data = data.drop(['sid', 'sample', 'sex1f', 'age', 'education_y', 'MMSE', 'MRI_field_strength',], axis=1)

# Divide float64 columns by 'eTIV'
float_columns = data.select_dtypes(include=['float64']).columns
data[float_columns] = data[float_columns].div(data['eTIV'], axis=0)
data = data.drop(['eTIV'], axis=1)

data['grp']= data['grp'].replace('CN', 0).replace('AD', 1).replace('MCI', 1)

# Separate target variable and input features
Y_val = data['grp']
X_val = data.drop(['grp'], axis=1)
scaler = StandardScaler()
X_val = scaler.fit_transform(X_val)

def fit_round(rnd: int) -> Dict:
    """Send round number to client"""
    return {"rnd": rnd}

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    x_val, y_val = X_val, Y_val
    
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        y_pred = model.predict(x_val)
        y_pred_labels = (y_pred >= 0.5).astype(int)
        y_true = y_val
        # Create lists to store y_pred and y_true
        y_pred_flat = y_pred_labels.ravel().tolist()
        y_true_list = y_val.tolist()
        y_true_flat = y_val.tolist()
        # Print y_pred and y_true for each round
        print(f"Round {server_round} - y_pred: {y_pred_flat}")
        print(f"Round {server_round} - y_true: {y_true_flat}")
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_labels)
        # Calculate precision, recall, and F1 score
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='weighted')
        # Print confusion matrix, precision, recall, and F1 score
        print(f"Confusion matrix:\n{cm}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 score: {f1_score:.4f}")
        # Print round number and completion message
        print(f"Round {server_round} evaluation completed")
        return loss, {"accuracy": accuracy}

    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 8,
        "local_epochs": 1,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

# Load and compile model for
# 1. server-side parameter initialization
# 2. server-side parameter evaluation
model = create_model()

# Create strategy
strategy = fl.server.strategy.FedAdagrad(
    min_fit_clients=4,
    min_evaluate_clients=4,
    min_available_clients=4,
    evaluate_fn=get_evaluate_fn(model),
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
)

# Start Flower server for federated learning
fl.server.start_server(
    server_address="0.0.0.0:9097",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
