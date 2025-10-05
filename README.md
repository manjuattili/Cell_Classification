# README: Cell Classification Project

# Mouse Brain Cell Type Classification

## Background

Neurons and glia are the two primary cell types in the brain. Neurons are specialized cells responsible for transmitting electrical and chemical signals, enabling functions like thinking, movement, and sensation. They are categorized into types such as principal cells (e.g., pyramidal neurons that form the backbone of neural circuits) and interneurons (which modulate and regulate neuronal activity). Glia, on the other hand, provide support to neurons, including maintenance of homeostasis, insulation (e.g., via myelin), and immune functions. Key glial types include astrocytes (which support synaptic function and blood-brain barrier) and microglia (the brain's immune cells that respond to injury and infection).

This project utilizes 3D reconstructions of these cells to extract morphometric features for classification. The data originates from NeuroMorpho.org, a comprehensive database of digitally reconstructed neurons and glia contributed by researchers worldwide. From NeuroMorpho.org, metadata and morphometric data for mouse brain cells were downloaded. Additionally, SWC files—standardized reconstruction files representing the 3D morphology of cells as a series of connected compartments—were obtained. As part of previous work, these SWC files were processed using geometry formulas to compute local and global angles around each bifurcation point, adding new columns to the dataset. This resulted in a dataset with 53 columns (including metadata, morphometric data, and computed angles) and 141,898 rows. For the classification task, only relevant numerical morphometric features were retained, reducing the dataset to 22 columns after preprocessing.


## Data
- **Source**: The raw dataset contains 141,896 rows and 52 columns, capturing characteristics of neurons and glia from the mouse brain, derived from 3D reconstructions available at [NeuroMorpho.org](https://neuromorpho.org).
- **Format**: Cell morphologies were downloaded as SWC files and compiled into a single dataset, with each row representing a single cell.

## Goal
The objective of this project is to:
- Perform **exploratory data analysis (EDA)** to understand the dataset.
- Use **machine learning algorithms** to classify cells into one of the 4 categories, 'principal', 'interneuron', 'astrocyte', or 'miscroglia', based on morphometric characteristics, such as cell size, number of neurites, number of bifurcations, and bifurcation angles.

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

### Feature Coefficients

```plaintext
                    astrocyte  interneuron   microglia  principal
bif_ampl_local     -0.076645    0.475145  -0.615458   0.216957
bif_ampl_remote    -0.754946   -0.107900   0.919246  -0.056400
branch_Order       -0.195749    0.020556   0.104588   0.070605
contraction         0.037888   -0.232588   0.041160   0.153540
depth              -5.381108   -1.150166   7.869781  -1.338507
diameter           -0.770627    0.014039   0.939310  -0.182723
eucDistance        -8.951089   10.261186 -14.741110  13.431013
fractal_Dim         0.146752   -0.147049  -0.156538   0.156835
fragmentation      -0.150478   -0.074094   0.290051  -0.065479
height            -15.621131   13.042885  -7.991352  10.569597
length              3.850072   -1.910787   0.753901  -2.693185
n_bifs              0.100680    0.125353  -0.152417  -0.073616
n_branch            0.097762    0.116343  -0.148725  -0.065381
n_stems            -0.202960   -0.077259   0.237907   0.042312
partition_asymmetry 0.267072   -0.207059   0.318207  -0.378220
pathDistance       -6.276812    4.272901  -2.412223   4.416134
pk_classic         -0.090134   -1.526386  -0.301553   1.918073
surface            -0.248590    0.240517  -0.857723   0.865796
volume              0.208882    0.841510  -2.118085   1.067692
width               8.746457   -6.709104   3.273382  -5.310735
local_mean          0.511505    0.188260  -0.083411  -0.616355
global_mean        -0.105660   -0.349910   0.063159   0.392410
```

- **High-Impact Features**: Features like height (e.g., -15.62 for astrocyte, 13.04 for interneuron), eucDistance (-8.95 for astrocyte, 13.43 for principal), and width (8.75 for astrocyte, -6.71 for interneuron) have large coefficients, indicating strong influence on class predictions.

## Conclusion
The Logistic Regression model achieves an accuracy of 82.45% but struggles with minority classes (interneuron, astrocyte) due to class imbalance. Using class weights or switching to other algorithms such as Random Forest or XGBoost could improve recall and F1-scores for these classes. Cross-validation and feature selection may further enhance model stability and performance.
