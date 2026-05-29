# ML Library

A simple, modular machine learning library built with scikit-learn.  
It includes models, evaluation metrics, and visualization tools for quick experimentation.

---

## 📂 Project Structure
ml_library/
├── models/
│   ├── linear.py
│   ├── tree.py
│   ├── svm.py
│   ├── forest.py
├── metrics/
│   ├── evaluation.py
├── visualization/
│   ├── plots.py
├── init.py
└── README.md


---

## 🚀 Usage Example

```python
import numpy as np
from ml_library import LinearModel, evaluate, plot_predictions

# Example dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearModel()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Evaluate
print(evaluate(y, y_pred))

# Visualize
plot_predictions(y, y_pred)
