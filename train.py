import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))

joblib.dump(model, "model.joblib")          # save
model = joblib.load("model.joblib")         # load

# ------------------------------------------------------------
# HOW TO RUN (Windows 11 - Command Prompt, inside this folder)
#   1. python -m venv venv
#   2. venv\Scripts\activate
#   3. pip install -r requirements.txt
#   4. python train.py        -> creates model.joblib
#   5. Then run the tests (see the comments at the end of test_model.py)
# Expected output: MSE: 2900.19...
# ------------------------------------------------------------
