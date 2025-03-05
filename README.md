# Federated Ensemble Learning of Neural Networks for MRI Data to Classify Dementia

## Overview

This repository contains the code and resources for my master's thesis, "Federated Ensemble Learning of Neural Networks for MRI Data to Classify Dementia." The goal of this project is to explore the use of federated learning techniques to improve the accuracy and privacy of dementia diagnosis using MRI data. Federated learning allows multiple institutions to collaboratively train a global model without sharing sensitive raw data, addressing privacy and data security concerns.

**Prepared By:** Aminah Dodiya

**Supervised By:**

*   Prof. Dr.-Ing. Thomas Kirste
*   Dr. rer. Hum. Martin Dyrba

**Cooperation with:** DZNE (German Center for Neurodegenerative Diseases)

## Problem Statement

Dementia is a growing health concern characterized by declining memory and cognitive abilities. Early and accurate diagnosis is crucial for effective treatment. Traditional diagnostic methods rely on manual processes and lack definitive markers. Medical imaging, particularly MRI, plays a critical role in dementia diagnosis. This thesis explores how artificial intelligence (AI) and machine learning can improve dementia detection using MRI data within a decentralized learning approach.

## Key Concepts

*   **Dementia:** Neurological disorders with declining memory and cognitive abilities.
*   **MRI:** Medical imaging technique crucial for dementia diagnosis.
*   **Federated Learning (FL):** A decentralized approach that enables collaborative model training without sharing raw data, preserving privacy.
*   **Federated Ensemble Learning (FEL):** Combines federated learning with ensemble learning principles to enhance predictive capabilities.

## Visualizing the Impact of Dementia

The image below illustrates the structural differences between a healthy brain and one Alzheimer's Disease progression.

![Healthy Brain vs. Dementia Brain](https://healthwasp.com/wp-content/uploads/the-stages-of-dementia4.jpg)

*Source:(https://www.weednews.co/cbd-oil-for-dementia-alzheimer/)*


## Repository Structure

The repository is organized into the following main folders:

*   `FL_with_mri_csv`: Contains code for federated learning experiments using MRI data in CSV format.
*   `FL_with_mri_images`: Contains code for federated learning experiments using MRI images directly.
*   `requirements.txt`: Lists the Python packages required to run the code.

### 1. federated\_with\_mri\_images

Each of the `FL_with_mri_csv/` and `FL_with_mri_images/` folders further contains subfolders for different Federated Learning strategies:

*   `FedAvg/`: Implementation of the FedAvg algorithm.
*   `FedAdam/`: Implementation of the FedAdam algorithm.
*   `weighted_FedAvg/`: Implementation of the Weighted FedAvg algorithm.
*   `FedAdagrad/`: Implementation of the FedAdagrad algorithm.
*   `centralized_learning/`: Implementation of Centralized Learning approach.
*   `test/`: Test scripts and data for evaluating the models.

Each strategy subfolder includes the following files:

*   `client1.py`, `client2.py`, `client3.py`, `client4.py`: Python scripts simulating the client-side training process.
*   `server.py`: Python script for the central server that orchestrates the Federated Learning process.
*   `data_preprocess.py`: Python script for preprocessing the MRI data.
*   `util.py`: Python script containing utility functions.
*   `run.sh`: A shell script to execute the experiment.

## Dependencies

To install the required packages, run the following command:

pip install -r requirements.txt


## How to Run the Experiments

Each subfolder in `federated_with_mri_csv` and `federated_with_mri_images` contains a `run.sh` script that automates the process of running the federated learning experiments. Follow these steps to run an experiment:

1.  Navigate to the directory of the experiment you want to run. For example:

    ```
    cd federated_with_mri_csv/binary_classification
    ```
Or
    ```
    cd federated_with_mri_images/FedAvg
    ```

2.  Make the `run.sh` script executable:

    ```
    chmod +x run.sh
    ```

3.  Run the script:

    ```
    ./run.sh
    ```

4.  Follow the instructions and the logs in the terminal

## Federated Learning Strategies

The following federated learning strategies were implemented and evaluated:

*   **FedAvg (Federated Averaging):** Focuses on global model by averaging local model updates through iterative communication rounds between clients and a central server.
*   **Weighted FedAvg:** A variant of FedAvg that assigns weights to local model updates based on the size or quality of the local datasets.
*   **FedAdam:** An optimization algorithm that combines federated learning with the Adam optimizer.
*   **FedAdagrad:** An optimization algorithm that combines federated learning with the Adagrad optimizer.

## Challenges and Limitations

*   Communication Overhead
*   Heterogeneous Data
*   Privacy Concerns
*   Model Synchronization
*   Sample Efficiency
*   Complexity and Resource Intensity
*   Algorithm Selection
*   Scalability                         |

## Datasets

The datasets used in this project were provided by The DZNE (German Center for Neurodegenerative Diseases). The datasets include MRI scans from four distinct study sources: ADNI2, ADNI3, AIBL, DELCODE, and EDSD. The datasets include three classes of participants: cognitively normal (CN), patients with (late) amnestic mild cognitive impairment (MCI), and patients with Alzheimer’s dementia (AD). Due to data privacy restrictions, the datasets themselves cannot be shared publicly.

## Model Architecture

A Convolutional Neural Network (CNN) model, specifically DenseNet, was used for this study.

## Acknowledgments

I would like to express my gratitude to Prof. Dr.-Ing. Thomas Kirste and Dr. rer. Hum. Martin Dyrba for their guidance and support throughout this project. I would also like to thank DZNE for providing the MRI datasets.

