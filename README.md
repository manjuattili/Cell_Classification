# README: Cell Classification Project

# Mouse Brain Cell Type Classification

## Background

Neurons and glia are the two primary cell types in the brain. Neurons are specialized cells responsible for transmitting electrical and chemical signals, enabling functions like thinking, movement, and sensation. They are categorized into types such as principal cells (e.g., pyramidal neurons that form the backbone of neural circuits) and interneurons (which modulate and regulate neuronal activity). Glia, on the other hand, provide support to neurons, including maintenance of homeostasis, insulation (e.g., via myelin), and immune functions. Key glial types include astrocytes (which support synaptic function and blood-brain barrier) and microglia (the brain's immune cells that respond to injury and infection).

This project utilizes 3D reconstructions of these cells to extract morphometric features for classification. The data originates from NeuroMorpho.org, a comprehensive database of digitally reconstructed neurons and glia contributed by researchers worldwide. From NeuroMorpho.org, metadata and morphometric data for mouse brain cells were downloaded. Additionally, SWC files—standardized reconstruction files representing the 3D morphology of cells as a series of connected compartments—were obtained. As part of previous work, these SWC files were processed using geometry formulas to compute local and global angles around each bifurcation point, adding new columns to the dataset. This resulted in a dataset with 53 columns (including metadata, morphometric data, and computed angles) and 141,898 rows. For the classification task, only relevant numerical morphometric features were retained, reducing the dataset to 22 columns after preprocessing.

## Overview

This project aims to classify cell types in the mouse brain using the processed morphometric data from 142,193 cells across 4 imbalanced classes: **astrocyte**, **interneuron**, **microglia**, and **principal**. Initial attempts focused on multiclass classification to distinguish all four types on the highly imbalanced dataset, achieving an accuracy of 82% with Logistic Regression. However, this approach struggled with class imbalance, performing better on majority classes like microglia and principal while underperforming on minorities like astrocytes and interneurons.

To improve results, the data was modified for binary classification (**Neuron: Yes/No**), combining neuronal types (interneuron and principal) into "Yes" and glial types (astrocyte and microglia) into "No". Four classification algorithms—Logistic Regression, Decision Tree, Random Forest, and XGBoost—were then applied with class balancing techniques, yielding significantly better performance, with XGBoost and Random Forest reaching ~99.7% accuracy.

Key highlights:
- **Dataset Size**: 135,538 cleaned entries.
- **Features**: 22 numerical morphometric features (e.g., `bif_ampl_local`, `diameter`, `eucDistance`).
- **Target**: Binary (`Neuron`: Yes/No) after modification.
- **Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost (with balanced weights).
- **Best Performance**: XGBoost and Random Forest achieve ~99.7% accuracy.

This repository contains a Jupyter notebook (`CP-EDA (1).ipynb`) with exploratory data analysis (EDA), preprocessing, model training, evaluation, and a generated report for model comparison.

## Dataset

- **Source**: NeuroMorpho.org (metadata, morphometrics, and SWC reconstruction files).
- **Original Classes**: 4 imbalanced classes (principal: 59,125; microglia: 56,237; interneuron: 15,980; astrocyte: 4,196).
- **Binary Modification**: Neuron (Yes: 75,105; No: 60,433).
- **Features**: Numerical morphometrics like bifurcation angles, branch order, Euclidean distance, fractal dimension, etc.
- **Preprocessing**:
  - Dropped columns with high missing values or irrelevant data (e.g., types 1-15, brain regions, domain, gender).
  - Removed rows with missing target or critical values.
  - Scaled features using `StandardScaler` post train-test split (80-20) to avoid leakage.
  - Encoded target with `LabelEncoder`.

## Data Cleaning and EDA
- **Data Examination**:
  - Data was examined to review data types of features, missingness, statistics.
  - Unwanted columns and missing rows were removed after careful consideration of whether the feature will help/make sense to be included in the model.
  - Data type for 'Type' was changed to 'category' from 'object'.
- **Target Variable (`Type`)**:
  - Target column, 'Type', was examined through bar plots.
  - It was found that the data is heavily imbalanced with principal cell counts close to 60,000 and astrocytes around 4,000 cells.
- **Feature Analysis**:
  - All numerical features were plotted for each class in 'Type' column to examine data distributions.
  - Upon eyeballing, it was noted that most features had distinct distributions.
  - Finally, a correlation map was plotted to understand relationships between the numerical features.
EDA includes:
- Summary statistics.
- Class distribution plots.
- Boxplots for features by class.
- Correlation heatmap.

## Multiclass Classification
## Model
- **Algorithm**: Logistic Regression with default parameters (no class weights) was used as a baseline model.
- **Features**:
  - Defined features (`X`) by excluding `Type`, `neuron_id`, and `neuron_name` (unique identifiers with no predictive value).
  - Total of 22 numerical features used (e.g., `bif_ampl_local`, `diameter`, `height`).
- **Preprocessing**:
  - Split dataset into 80% training and 20% test sets (~27,108 test samples).
  - Scaled numerical features after the split to prevent data leakage.
- **Training and Evaluation**:
  - Trained the Logistic Regression model and made predictions.
  - Evaluated using accuracy score, classification report, confusion matrix, and feature coefficients per class.

## Model Performance

### Class Distribution
The test set class distribution is:
- `principal`: 11,754 (~43.4%)
- `microglia`: 11,355 (~41.9%)
- `interneuron`: 3,121 (~11.5%)
- `astrocyte`: 878 (~3.2%)
This indicates significant class imbalance, with `principal` and `microglia` dominating, while `astrocyte` and `interneuron` are minority classes, particularly `astrocyte`.

### Accuracy Score
- **Accuracy**: 0.8245 (82.45%)
  - The model correctly predicts 82.45% of test instances (27,108 samples).
  - Accuracy is skewed by the majority classes (`principal` and `microglia`), which account for ~85.3% of the test set.

### Classification Report
The classification report provides precision, recall, and F1-score for each class:

```plaintext
              precision    recall  f1-score   support
astrocyte        0.77      0.45      0.57       878
interneuron      0.53      0.08      0.14      3121
microglia        0.89      0.96      0.93     11355
principal        0.78      0.92      0.84     11754
accuracy                           0.82     27108
macro avg        0.74      0.60      0.62     27108
weighted avg     0.80      0.82      0.79     27108
```

- **Per-Class Analysis**:
  - **astrocyte**: Moderate precision (0.77) but low recall (0.45), indicating only 45% of `astrocyte` instances are correctly identified, with a low F1-score (0.57) due to imbalance.
  - **interneuron**: Poor precision (0.53) and extremely low recall (0.08), resulting in a very low F1-score (0.14). The model struggles significantly with this class, likely due to its minority status.
  - **microglia**: High precision (0.89), recall (0.96), and F1-score (0.93), reflecting strong performance for this majority class.
  - **principal**: Good precision (0.78), high recall (0.92), and solid F1-score (0.84), benefiting from its large support.
- **Macro Average**: Precision (0.74), recall (0.60), F1-score (0.62) indicate uneven performance, possible due to imbalance of classes.
- **Weighted Average**: Precision (0.80), recall (0.82), F1-score (0.79) are closer to accuracy, reflecting the dominance of majority classes.

### Confusion Matrix
The confusion matrix shows the number of true vs. predicted labels for each class:

```plaintext
            astrocyte  interneuron  microglia  principal
astrocyte        394          1        415         68
interneuron        2        242        213       2664
microglia         65          4      10952        334
principal         53        212        727      10762
```

- **Accurate Predictions**:
  - `astrocyte`: 394
  - `interneuron`: 242
  - `microglia`: 10,952
  - `principal`: 10,762
- **Analysis**: Low recall for `astrocyte` (415 misclassified as `microglia`) and `interneuron` (2,664 misclassified as `principal`), indicating difficulty distinguishing minority classes from majority ones.

- **High-Impact Features**: Feature coefficients were extracted. It was observed that features like height (e.g., -15.62 for astrocyte, 13.04 for interneuron), eucDistance (-8.95 for astrocyte, 13.43 for principal), and width (8.75 for astrocyte, -6.71 for interneuron) have large coefficients, indicating strong influence on class predictions.

## Conclusion
The Logistic Regression model achieves an accuracy of 82.45% but struggles with minority classes (interneuron, astrocyte) due to class imbalance. Using class weights or switching to other algorithms such as Random Forest or XGBoost could improve recall and F1-scores for these classes. Cross-validation and feature selection may further enhance model stability and performance.

## Binary Classification
## Methods

  - Models trained with class balancing (`class_weight='balanced'` or `scale_pos_weight` for XGBoost) to handle the mild imbalance in the binary setup.
  - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- **Feature Importance**: Extracted and visualized for each model (absolute coefficients for Logistic Regression; gain-based for others).

## Results

### Model Comparison Report

Dataset Overview: Modified dataset with 135,538 entries, no missing values. Target: `Neuron` (binary: `Yes` for `interneuron` and `principal`, `No` for `astrocyte` and `microglia`). Features: 22 numerical features. Preprocessing: Scaled numerical features after 80-20 train-test split. Models: Logistic Regression, Decision Tree, Random Forest, XGBoost (all with balanced class weights or equivalent).

Class Distribution:

Neuron  
Yes    75105  
No     60433  

Model Performance:

| Model                | Accuracy | TP     | FP   | TN     | FN   | Precision (Yes) | Recall (Yes) | F1 (Yes) |
|----------------------|----------|--------|------|--------|------|-----------------|--------------|----------|
| Logistic Regression  | 0.9463  | 13794 | 376 | 11857 | 1081| 0.97           | 0.93        | 0.95    |
| Decision Tree        | 0.9871  | 14687 | 162 | 12071 | 188 | 0.99           | 0.99        | 0.99    |
| Random Forest        | 0.9956  | 14822 | 65  | 12168 | 53  | 1.00           | 1.00        | 1.00    |
| XGBoost              | 0.9972  | 14832 | 32  | 12201 | 43  | 1.00           | 1.00        | 1.00    |

- **Top Features**: `eucDistance`, `height`, `pathDistance`, `width` consistently rank high across models.
- **Visualizations**: Class distributions, boxplots, correlation heatmap, feature importance bars.

Multiclass Results (Logistic Regression): ~82% accuracy; better for majority classes (microglia: 93% F1, principal: 84% F1).

### Top 3 Features by Model

| Model                | Rank 1 (Importance)     | Rank 2 (Importance)   | Rank 3 (Importance)   |
|----------------------|-------------------------|-----------------------|-----------------------|
| Logistic Regression  | eucDistance (1.0000)   | height (0.8438)      | width (0.4239)       |
| Decision Tree        | eucDistance (0.7903)   | n_bifs (0.0328)      | diameter (0.0236)    |
| Random Forest        | height (0.1999)        | eucDistance (0.1968) | pathDistance (0.1587)|
| XGBoost              | eucDistance (0.7677)   | n_bifs (0.0458)      | height (0.0212)      |

## Conclusion

This project demonstrates the efficacy of machine learning in classifying mouse brain cell types based on morphometric features derived from 3D reconstructions. The initial multiclass approach highlighted challenges with class imbalance, achieving only 82% accuracy with Logistic Regression, underscoring the difficulty in distinguishing fine-grained subtypes like astrocytes and interneurons. By reframing the problem as binary classification between neurons and glia, performance improved dramatically, with ensemble methods like Random Forest and XGBoost attaining near-perfect accuracy (~99.7%). This suggests that morphometric differences are more pronounced at the neuron-glia level, making binary classification a more robust strategy for such datasets.

Key insights include the prominence of spatial and structural features such as eucDistance, height, and pathDistance across models, which likely capture the elongated, arborized morphology of neurons versus the more compact structure of glia. These features—along with other top-ranked ones like n_bifs, diameter, and width—are derived from standard neuromorphological metrics used in analyzing 3D reconstructions of brain cells (e.g., via tools like L-Measure from NeuroMorpho.org). They quantify aspects of cellular geometry, branching, and spatial extent, which differ markedly between neurons and glia due to their functional roles. For instance, eucDistance (Euclidean Distance) measures the maximum straight-line distance from the soma to the farthest tip of the cellular arbor, reflecting neurons' extensive dendritic and axonal spans for signal integration over distances, while glia remain localized. Similarly, height captures the vertical extent along a predefined axis, highlighting neurons' apical-basal polarity (e.g., in pyramidal cells), and pathDistance quantifies the longest tortuous path along branches, emphasizing neurons' winding arbors for connectivity versus glia's shorter processes.

Furthermore, n_bifs (Number of Bifurcations) counts branch points, indicating neurons' higher branching complexity for synaptic surface area; diameter represents branch thickness, with neurons often having varied calibers for efficient conduction compared to glia's finer extensions; and width measures lateral spread, underscoring neurons' broad dendritic fields for input sampling. Collectively, these features highlight how neurons' morphology is adapted for information processing and transmission over distances, resulting in larger spatial extents and greater complexity, while glia prioritize local interactions with simpler structures. This aligns with evolutionary pressures: neurons form expansive networks for cognition, whereas glia provide on-site support like nutrient delivery (astrocytes) or debris clearance (microglia). The models' reliance on these metrics suggests that simple geometric descriptors from 3D reconstructions can serve as robust biomarkers for automated classification, potentially accelerating neuroscience research by enabling high-throughput analysis of cell types in brain atlases. However, variations within classes (e.g., bushy vs. elongated astrocytes) could explain residual errors, emphasizing the need for context-aware interpretations in biological applications. These findings have implications for neuroscience, potentially aiding automated cell type identification in large-scale brain mapping efforts. Future work could explore incorporating additional features from SWC files, addressing remaining imbalances with advanced techniques like SMOTE, or applying deep learning algorithms on a larger dataset.

## Acknowledgments

- Data sourced from NeuroMorpho.org.
- Built with scikit-learn and XGBoost for modeling.






