import joblib
import numpy as np

model = joblib.load("model.joblib")

def test_prediction_type():
    pred = model.predict(np.zeros((1, 10)))
    assert isinstance(pred[0], float)

def test_output_shape():
    pred = model.predict(np.zeros((5, 10)))
    assert pred.shape == (5,)

# ------------------------------------------------------------
# HOW TO RUN (Windows 11 - Command Prompt, inside this folder)
#   1. python -m venv venv
#   2. venv\Scripts\activate
#   3. pip install -r requirements.txt
#   4. python train.py              -> creates model.joblib
#   5. pytest test_model.py -v
# Expected output: 2 passed
#
# GitHub Actions (the workflow file is .github\workflows\ci.yml):
#   Create an empty repo on github.com, then in this folder:
#     git init
#     git add .
#     git commit -m "Add CI pipeline"
#     git branch -M main
#     git remote add origin https://github.com/<user>/<repo>.git
#     git push -u origin main
#   Open the repo on GitHub -> Actions tab to watch the jobs run.
#   (Do not push the venv folder; .gitignore in this folder excludes it.)
# ------------------------------------------------------------
