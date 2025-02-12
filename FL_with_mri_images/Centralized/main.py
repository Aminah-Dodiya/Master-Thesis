import available_datasets
import numpy as np
import keras
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split 
import random
import util
import datetime

# Set random seeds
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
tf.compat.v1.set_random_seed(0)
keras.backend.clear_session()


#flavor = 'N4cor_Warped_bet'
flavor = 'N4cor_WarpedSegmentationPosteriors2'
adni2 = available_datasets.load_ADNI2_data(x_range_from = 32, x_range_to = 161,
    y_range_from = 90, y_range_to = 135,
    z_range_from = 20, z_range_to = 159, flavor = flavor)
aibl = available_datasets.load_AIBL_data(x_range_from = 32, x_range_to = 161, 
    y_range_from = 90, y_range_to = 135,
    z_range_from = 20, z_range_to = 159, flavor = flavor)
edsd = available_datasets.load_EDSD_data(x_range_from = 32, x_range_to = 161, 
    y_range_from = 90, y_range_to = 135,
    z_range_from = 20, z_range_to = 159, flavor = flavor)
delcode = available_datasets.load_DELCODE_data(x_range_from = 32, x_range_to = 161, 
    y_range_from = 90, y_range_to = 135,
    z_range_from = 20, z_range_to = 159, flavor = flavor)
adni3 = available_datasets.load_ADNI3_data(x_range_from = 32, x_range_to = 161, 
    y_range_from = 90, y_range_to = 135,
    z_range_from = 20, z_range_to = 159, flavor = flavor)

# combine datasets
images = np.concatenate([ adni2['images'], aibl['images'], edsd['images'], delcode['images'] ], axis=0)
labels = np.concatenate([ adni2['labels'], aibl['labels'], edsd['labels'], delcode['labels'] ], axis=0)
labels = to_categorical(labels)
groups = np.concatenate([ adni2['groups'], aibl['groups'], edsd['groups'], delcode['groups'] ], axis=0)
covariates = np.concatenate([ adni2['covariates'], aibl['covariates'], edsd['covariates'], delcode['covariates'] ], axis=0)
numfiles = labels.shape[0]

# Randomly select a subset of the data based on the specified percentage
#num_samples = int(images.shape[0] * 0.001)
#random_indices = np.random.choice(images.shape[0], num_samples, replace=False)
#images_subset = images[random_indices]
#labels_subset = labels[random_indices]
#groups_subset = groups[random_indices]
#covariates_subset = covariates[random_indices]
#numfiles = labels.shape[0]

x_val = adni3['images']
y_val = adni3['labels']
y_val = to_categorical(y_val)

#X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.001, stratify = labels, random_state=42)
#print("Training Input shape\t: {}".format(X_train.shape))
#print("Testing Input shape\t: {}".format(X_test.shape))
#print("Training Output shape\t: {}".format(Y_train.shape))
#print("Testing Output shape\t: {}".format(Y_test.shape))

# define class weights to train a balanced model, taken from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
neg, pos = np.bincount(labels[:, 1].astype(np.int_))
weight_for_0 = (1 / neg)*(numfiles)/2.0
weight_for_1 = (1 / pos)*(numfiles)/2.0
class_weights = {0: weight_for_0, 1: weight_for_1}
print('Examples:    Total: {}    Positive: {} ({:.2f}% of total)'.format(
    numfiles, pos, 100 * pos / numfiles))
print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# Load and compile Keras model
model = util.DenseNet(images.shape[1:], labels.shape[1])

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])



# Fit model to training data
batch_size = 8
epochs = 25
hist = model.fit(images, labels,
                        batch_size=batch_size,
                        epochs=epochs, 
                        verbose=1,
                        validation_split=0.1,
                        class_weight=class_weights)


mymodel = hist.model
# Calculate accuracy for holdout test data
scores = mymodel.evaluate(x_val, y_val, batch_size=batch_size) #, verbose=0
print("Test %s: %.2f%%" % (mymodel.metrics_names[1], scores[1]*100))


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
print("y_pred:", y_pred_list)
print("y_true:", y_true_list)
# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Calculate precision, recall, and F1 score
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
# Print confusion matrix, precision, recall, and F1 score
print(f"Confusion matrix:\n{cm}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1_score:.4f}")
print("Test %s: %.2f%%" % (mymodel.metrics_names[1], scores[1]*100))

# Get the current time
end_time = datetime.datetime.now()

# Print the end time
print("End time:", end_time)
