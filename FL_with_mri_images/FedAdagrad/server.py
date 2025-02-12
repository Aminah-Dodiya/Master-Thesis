# import required libraries
import glob
import re
import numpy as np
import flwr as fl
import available_datasets
import keras
from keras.utils import to_categorical
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, roc_auc_score, auc
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
adni3 = available_datasets.load_ADNI3_data(x_range_from = 32, x_range_to = 161,
                                           y_range_from = 90, y_range_to = 135, 
                                           z_range_from = 20, z_range_to = 159, 
                                           flavor = flavor)
adni3_images = adni3['images']
adni3_labels = adni3['labels']
adni3_labels = to_categorical(adni3_labels)

# Split the data into training and testing set
print("Validation Input shape\t: {}".format(adni3_images.shape))
print("validation Output shape\t: {}".format(adni3_labels.shape))

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    x_val, y_val = adni3_images, adni3_labels

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Update model with the latest parameters
        model.set_weights(parameters)  
        y_pred = model.predict(x_val)
        
        y_pred_prob = y_pred
        # Compute ROC curve and area
        tpr_list =[] 
        fpr_list =[] 
        roc_auc_list =[]
        for class_index in range(y_val.shape[1]):
            fpr, tpr, _ = roc_curve(y_val[:, class_index], y_pred_prob[:, class_index])
            roc_auc = roc_auc_score(y_val[:, class_index], y_pred_prob[:, class_index])
            fpr_list.append(fpr.tolist())
            tpr_list.append(tpr.tolist())
            roc_auc_list.append(roc_auc)
        print('fpr:', fpr_list)
        print('tpr:', tpr_list)
        print('roc_auc:', roc_auc_list)

        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_val, axis=1)
        # Create lists to store y_pred and y_true
        y_pred_list = y_pred.tolist()
        y_true_list = y_true.tolist()
        # Print y_pred and y_true for each round
        print(f"Round {server_round} - y_pred: {y_pred_list}")
        print(f"Round {server_round} - y_true: {y_true_list}")
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Calculate precision, recall, and F1 score
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        # Print confusion matrix, precision, recall, and F1 score
        print(f"Confusion matrix:\n{cm}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 score: {f1_score:.4f}")
        loss, accuracy = model.evaluate(x_val, y_val)
        # Print round number and completion message
        print(f"Round {server_round} evaluation completed")
        return loss, {"accuracy": accuracy}

    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.
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
model = util.DenseNet(adni3_images.shape[1:], adni3_labels.shape[1])

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
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)

