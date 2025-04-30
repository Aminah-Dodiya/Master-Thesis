# Federated Ensemble Learning of Neural Networks for MRI Data to Classify Dementia

**Author:** Aminah Dodiya  
**Supervisors:** Prof. Dr.-Ing. Thomas Kirste, Dr. rer. Hum. Martin Dyrba  
**Affiliations:** University of Rostock, Deutsches Zentrum Für Neurodegenerative Erkrankungen e.V. (DZNE)  

---

## Overview

This repository contains the implementation and experimental framework for my master’s thesis:  **Federated Ensemble Learning (FEL)** of Neural Networks for MRI Data to Classify Dementia. The project addresses challenges of privacy, data heterogeneity, and limited data availability by enabling decentralized collaborative learning across multiple healthcare providers without sharing sensitive patient data.

---

## Motivation
Dementia diagnosis from MRI data is a complex challenge, often hindered by small, siloed datasets and strict privacy regulations. Federated Ensemble Learning (FEL) facilitates robust, privacy-preserving model training by aggregating knowledge from distributed sources, improving diagnostic accuracy while maintaining compliance with data protection regulations. This project demonstrates the feasibility and effectiveness of FEL for robust, scalable, and privacy-preserving dementia classification.

---

## Project Highlights

- Developed a federated ensemble learning framework simulating real-world hospital collaboration.
- Implemented and compared multiple aggregation strategies: FedAvg, FedAdam, Weighted FedAvg, and FedAdagrad.
- Employed a **customized DenseNet architecture** optimized for 3D MRI data ([Singh & Dyrba, 2023](https://doi.org/10.1007/978-3-658-41657-7_51)).
- Utilized the [Flower](https://flower.dev/) federated learning framework for scalable, flexible experimentation.

---

## Visual Summary

The image below illustrates the structural differences between a healthy brain and one affected by dementia, highlighting the importance of advanced imaging and analysis for early diagnosis.

![Healthy Brain vs. Dementia Brain](https://healthwasp.com/wp-content/uploads/the-stages-of-dementia4.jpg)

*Source:(https://www.weednews.co/cbd-oil-for-dementia-alzheimer/)*

---

## Repository Structure

```
Master-Thesis/
├── FL_with_mri_csv/
│   ├── FedAvg/
│   ├── FedAdam/
│   ├── weighted_FedAvg/
│   ├── FedAdagrad/
│   ├── centralized_learning/
│   └── test/
├── FL_with_mri_images/
│   ├── FedAvg/
│   ├── FedAdam/
│   ├── weighted_FedAvg/
│   ├── FedAdagrad/
│   ├── centralized_learning/
│   └── test/
├── report
│   └── thesis_report.pdf
├── LICENSE
├── README.md
└── requirements.txt
```
Each strategy folder contains:  
- `client1.py` – `client4.py`: Simulated client training scripts  
- `server.py`: Central server aggregation logic  
- `data_preprocess.py`: Data preprocessing utilities  
- `util.py`: Helper functions  
- `run.sh`: Shell script to run experiments
- `<strategy>_log/` (e.g., `FedAdam_log/`): Directory containing logs from all participants in the experiment:
  - `client1.log`, `client2.log`, `client3.log`, `client4.log`: Logs capturing each client’s local training process.
  - `server.log`: Logs detailing server-side coordination, aggregation, and communication events.

This organized logging structure facilitates:
- Detailed debugging of training and aggregation steps.
- Performance monitoring for each client and the server.
- Reproducibility and transparency of all federated learning experiments.

---

## Data and Preprocessing

- **Datasets:** Multi-institutional MRI datasets from ADNI2, AIBL, DELCODE, and EDSD, including three classes: cognitively normal (CN), mild cognitive impairment (MCI), and Alzheimer’s dementia (AD).
- **Preprocessing:** Includes normalization, standardization, and augmentation to improve model robustness.
- **Data Partitioning:** Simulates non-IID distribution reflecting real-world hospital data heterogeneity.
- **Privacy:** Raw data remains local on client nodes, ensuring compliance with GDPR and HIPAA.

---

## Model Architecture

The project uses a **customized DenseNet** architecture adapted for volumetric MRI data:

- Dense connectivity between layers to improve feature reuse and gradient flow.
- 3D convolutional layers to capture spatial features in MRI volumes.
- Transition layers to reduce feature map size and computational complexity.
- Output layer with softmax activation for multi-class classification (CN, MCI, AD).

*Reference: Singh, A., & Dyrba, M. (2023). Comparison of CNN Architectures for Detecting Alzheimer’s Disease using Relevance Maps. In Bildverarbeitung für die Medizin 2023 (BVM 2023). DOI: [10.1007/978-3-658-41657-7_51](https://doi.org/10.1007/978-3-658-41657-7_51)*

---

## Federated Learning Implementation

Federated learning experiments are implemented using the [**Flower**](https://flower.dev/) framework, which provides:

- Scalable orchestration of decentralized training.
- Support for custom aggregation strategies.
- Seamless integration with PyTorch and TensorFlow.

**Why Flower?**  
Flower enables production-grade federated learning with flexible client-server architecture, making it ideal for research and deployment.  
[Flower Documentation](https://flower.dev/docs/)

**Aggregation Strategies Implemented:**

- **FedAvg:** Standard federated averaging.  
- **FedAdam:** Federated adaptive moment estimation.  
- **Weighted FedAvg:** Aggregation weighted by client dataset size.  
- **FedAdagrad:** Federated Adagrad optimizer.  
- **Centralized Learning:** Baseline using pooled data for comparison.

---

## Experimental Setup

- **Simulation:** Four clients representing distinct healthcare datasets, coordinated by a central server.
- **Evaluation Metrics:** Accuracy, ROC-AUC, loss curves, and confusion matrices.
- **Reproducibility:** All experiments are reproducible via provided scripts and documented dependencies.

---

## Dependencies

To install the required packages, run the following command:

pip install -r requirements.txt

---

## How to Run

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Navigate to a strategy directory, for example:
    ```
    cd FL_with_mri_csv/FedAvg/
    ```
3. Execute the experiment:
    ```
    ./run.sh
    ```
Repeat for other strategies or for the `FL_with_mri_images` folder.

---

## Results

| Strategy         | Accuracy | ROC-AUC | Notes                     |
|------------------|----------|---------|---------------------------|
| Weighted FedAvg  | 0.89     | 0.91    | Best federated performance|
| FedAvg           | 0.87     | 0.89    | Competitive                |
| FedAdagrad       | 0.86     | 0.88    | Moderate improvement       |
| FedAdam          | 0.83     | 0.85    | Lower performance          |
| Centralized      | 0.90     | 0.92    | Upper bound (pooled data)  |

- Weighted FedAvg closely matches centralized learning performance while preserving privacy.
- FedAdam’s lower performance is attributed to sensitivity to non-IID data distributions.
- Federated ensemble learning effectively balances privacy and accuracy.

*For detailed results and analysis, see the [thesis report](report/thesis_report.pdf).*

---

## Future Work

- Explore advanced privacy techniques such as differential privacy and secure multiparty computation.
- Extend to other medical imaging modalities and diseases.
- Deploy in real clinical environments with live data streams.

---

## Acknowledgments

This thesis was conducted under the supervision of Prof. Dr.-Ing. Thomas Kirste and Dr. rer. Hum. Martin Dyrba, in collaboration with the German Center for Neurodegenerative Diseases (DZNE) and the University of Rostock. Special thanks to all data providers and collaborators.

---

## License

Distributed under the MIT License. See LICENSE for details.

---

**For more information, please refer to the full thesis report ([thesis_report.pdf](report/thesis_report.pdf)) or contact the author.**
