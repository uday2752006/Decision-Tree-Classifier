# 🌸 Iris Classification using Decision Tree and Random Forest

##  Objective
Learn and implement **tree-based models** for **classification** using the Iris dataset. Understand how to train, visualize, and evaluate models like **Decision Trees** and **Random Forests**, and interpret model behavior.

---

##  Tools & Libraries
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Graphviz (via `plot_tree` for visualization)

---

##  Dataset
The classic **Iris dataset** is used, which contains:
- 150 rows, 4 input features (sepal length/width, petal length/width), and 1 output class (species).
- Target classes: `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`

---

##  Project Steps

### 1. Data Loading and Exploration
- Read the dataset using `pandas`
- Explore using `.info()` and `.describe()`

### 2. Data Splitting
- Split into training and testing sets using `train_test_split`

### 3. Decision Tree Classifier
- Train a `DecisionTreeClassifier`
- Control **tree depth** with `max_depth` to prevent overfitting
- Visualize the tree using `plot_tree`

### 4. Random Forest Classifier
- Train a `RandomForestClassifier` with 100 estimators
- Compare accuracy with the Decision Tree model

### 5. Feature Importance
- Plot feature importances using `feature_importances_` and Seaborn

### 6. Model Evaluation
- Evaluate with:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

### 7. Cross-Validation
- Use `cross_val_score` to perform 5-fold cross-validation on the Decision Tree

---

##  Results
- Both models achieved high accuracy
- Random Forest generally performs better due to ensemble averaging
- Petal length and petal width were found to be the most important features

---

##  Visualization Samples
- Decision Tree structure
- Feature importance bar plots
- Confusion matrix (optional heatmap)

---

##  Future Improvements
- Add regression examples with `DecisionTreeRegressor`
- Implement GUI using Streamlit or Tkinter
- Hyperparameter tuning with GridSearchCV

---

##  Author
Developed as a learning project to understand tree-based classification models using Scikit-learn.

---

