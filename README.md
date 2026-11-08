# VinoClass AI
## Physicochemical Wine Quality Analytics & Machine Learning Benchmarks
## Developed By: APPANA MUKESH
## BITS ID: 2026AJ05910

### Machine Learning Classification Benchmark & Streamlit Deployment

## a. Problem Statement
The objective of this project is to build, evaluate, and benchmark five distinct supervised machine learning classification algorithms to predict wine quality (binary classification: High Quality vs Regular Quality). The trained models are deployed via an interactive Streamlit web application on Streamlit Community Cloud for automated inference and metric evaluation on unseen test datasets.

## b. Dataset Description
- **Dataset:** UCI Red Wine Quality Dataset
- **Instances:** 1,599 observations (Exceeds minimum requirement of 500)
- **Features:** 11 continuous physicochemical features + 1 target variable (Total 12 attributes):
  1. `fixed acidity`
  2. `volatile acidity`
  3. `citric acid`
  4. `residual sugar`
  5. `chlorides`
  6. `free sulfur dioxide`
  7. `total sulfur dioxide`
  8. `density`
  9. `pH`
  10. `sulphates`
  11. `alcohol`
- **Target Variable:** `target` (Binary encoded: `1` for quality >= 7 indicating high quality wine, `0` for quality < 7 indicating standard quality).

## c. GitHub Repository Link
- Repository URL: `https://github.com/arrjvvmukesh/BITS_ML_Assignment_2_Wine_ML_Classifier_VinoClass_AI_2026AJ05810.git`

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8938 | 0.8804 | 0.6957 | 0.3721 | 0.4848 | 0.4580 |
| **Decision Tree** | 0.9187 | 0.8611 | 0.7073 | 0.6744 | 0.6905 | 0.6440 |
| **KNN** | 0.8906 | 0.8215 | 0.6429 | 0.4186 | 0.5070 | 0.4617 |
| **Naive Bayes** | 0.8594 | 0.8517 | 0.4844 | 0.7209 | 0.5794 | 0.5131 |
| **Support Vector Machine** | 0.9000 | 0.8892 | 0.7619 | 0.3721 | 0.5000 | 0.4876 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivers strong discrimination with an AUC of 0.8804 and solid overall accuracy (89.38%). However, constrained by a linear boundary, recall is restricted to 37.21% due to positive class rarity. |
| **Decision Tree** | Outstanding overall balance on tabular data. Captures nonlinear threshold relationships effectively, yielding the highest accuracy (91.87%), top F1 score (0.6905), and highest MCC (0.6440) across all classifiers. |
| **KNN** | Demonstrates solid performance with an accuracy of 89.06% and F1 score of 0.5070 after feature standardization, though neighborhood-based classification yields the lowest AUC (0.8215) in this feature space. |
| **Naive Bayes** | Achieves the highest recall across all models (72.09%), effectively flagging positive wine samples. However, due to feature independence assumptions, precision drops to 48.44%, resulting in more false positives. |
| **Support Vector Machine** | Yields the highest precision (76.19%) and the best AUC score (0.8892) via the RBF kernel, maintaining 90.00% accuracy. However, positive instance detection (recall: 37.21%) remains conservative. |
| **Overall, Winner for your dataset?** | **Decision Tree** is the overall winner for this dataset. It achieves the highest overall accuracy (91.87%), superior balance between precision (70.73%) and recall (67.44%), leading to the best F1 score (0.6905) and Matthews Correlation Coefficient (0.6440). |

## e. How to Run Locally
1. Clone the repository:
   - git clone [https://github.com/arrjvvmukesh/BITS_ML_Assignment_2_Wine_ML_Classifier_VinoClass_AI_2026AJ05810.git](https://github.com/arrjvvmukesh/BITS_ML_Assignment_2_Wine_ML_Classifier_VinoClass_AI_2026AJ05810.git)
  -  cd BITS_ML_Assignment_2_Wine_ML_Classifier_VinoClass_AI_2026AJ05810

2. Verify Python Version (3.12):
   -python --version
   (Expected output: Python 3.12.x)

3. Create and Activate Virtual Environment:
   
  Linux / macOS / BITS Virtual Lab:
    -python3.12 -m venv venv
    -source venv/bin/activate
   
  Windows (Command Prompt):
    -py -3.12 -m venv venv
    -venv\Scripts\activate

  Windows (PowerShell):
    - py -3.12 -m venv venv
    -.\venv\Scripts\Activate.ps1

4. Install Dependencies:
  -python -m pip install --upgrade pip (Option If in case you encounter any issues/errors related to package installation, pip etc..)
  -pip install -r requirements.txt
  
5. Train Models and Generate Artifacts:
  -python train_and_export.py
  
6. Run the Streamlit Application Locally:
  -streamlit run app.py
  
7. Streamlit Cloud Deployment Guide:
  -For Streamlit Cloud Deployment refer this link and navigate to Docs for Deployment [https://streamlit.io/cloud](https://streamlit.io/cloud)

## Streamlit Deployed App Link: [https://mukesh-2026aj05810-ml-assignment-2-vinoclass-ai.streamlit.app/](https://mukesh-2026aj05810-ml-assignment-2-vinoclass-ai.streamlit.app/)
