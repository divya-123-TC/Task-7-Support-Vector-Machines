# Task-7-Support-Vector-Machines


Support Vector Machine (SVM) Classification – AI & ML Internship Task

Objective:

To implement SVM (Support Vector Machine) for binary classification using a synthetic dataset. The goal is to:

Train models using Linear and RBF kernels

Visualize decision boundaries

Tune hyperparameters using GridSearchCV

Evaluate model performance on test data



Dataset Used:

Generated using sklearn.datasets.make_classification

2 features (for 2D decision boundary visualization)

300 samples, binary classification (y ∈ {0, 1})





Tools & Libraries:

Python

Scikit-learn

NumPy

Matplotlib



Steps Performed:

 Step 1: Dataset Creation

Used make_classification() to generate a synthetic dataset with:

2 informative features

No redundant features

Separated class clusters



Step 2–3: Train/Test Split & Normalization

Split dataset into 70% training and 30% testing

Standardized the features using StandardScaler


Step 4–5: Model Training

Trained Linear SVM using kernel='linear'

Trained RBF (non-linear) SVM using kernel='rbf'


Step 6–7: Visualization

Plotted decision boundaries for both Linear and RBF models using contourf

Visualized how SVM separates the classes in 2D feature space


step 8: Hyperparameter Tuning

Used GridSearchCV to tune C and gamma for RBF SVM

Performed 5-fold cross-validation

Displayed best parameters and best CV score


Step 9: Evaluation

Evaluated best SVM model on test set

Printed accuracy score





Sample Output:

Best Parameters: {'C': 10, 'gamma': 1}
Best Cross-Validation Score: 0.99
Test Accuracy: 0.9667

 Visuals:

Linear SVM Decision Boundary

RBF Kernel SVM Decision Boundary








