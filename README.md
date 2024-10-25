# Loan-Approval-Prediction-using-Convolutional-Neural-Network-CNN

## Project Overview
This project aims to predict loan approval status based on applicant data using a Convolutional Neural Network (CNN) model. The goal is to develop a reliable machine learning model capable of accurately forecasting loan approval status, assisting financial institutions in assessing potential risk.

## Dataset
This project uses a dataset containing loan applicant information, divided into training and test sets:

- **Training set:** Contains applicant information and loan approval status labels (loan_status).

- **Test set:** Contains similar information without the loan_status column, used for making predictions.

Each entry includes features like:

- Personal Details (e.g., person_home_ownership, cb_person_default_on_file)
- Loan Details (e.g., loan_intent, loan_grade)
- Financial Information (e.g., income, debt, credit score)

## Data Preprocessing

- **Handle Missing Values:** Missing values in categorical and numerical columns are filled with the mode and mean, respectively.
- **Encoding Categorical Variables:** Encode categorical features using numerical mappings to make them compatible with the neural network.
- **Standardization:** Scale numerical features using StandardScaler to improve model performance and convergence.
- **Reshape Data:** Reshape the input data to be compatible with the CNN model.

## Model Architecture
The Convolutional Neural Network (CNN) used in this project has the following architecture:

- **Input Layer:** Reshaped features
- **Conv1D Layers:** Convolutional layers to extract meaningful patterns
- **Dropout Layers:** Regularization to prevent overfitting
- **Dense Layers:** Fully connected layers for classification

## Training and Evaluation
The model is trained on the training dataset with the following parameters:

- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Area Under the Curve (AUC), Accuracy

## Prediction
The model predicts loan approval status on the test dataset, and the results are saved as a binary outcome (1 for approval, 0 for rejection) in predictions.csv.

## Results
Key performance metrics of the model:

- Accuracy: 94.91%
- AUC: 93.37%

For a visual representation, actual vs. predicted values are plotted to compare model performance.

![image](https://github.com/user-attachments/assets/948dee11-fe31-47b3-9288-ff2f949d125f)

## Conclusion
In this project, we developed a CNN-based model to predict loan approval status using historical applicant data. By preprocessing the dataset and transforming categorical variables into numeric format, we ensured compatibility with the CNN model, which allowed us to capture complex patterns within the data. Our approach demonstrated that deep learning, particularly CNN, can be an effective method for financial risk prediction when applied to tabular data with appropriate feature engineering and model architecture.

Key findings and achievements include:

- **Improved Accuracy and AUC:** The model achieved significant predictive performance, reflecting its ability to discern the underlying patterns associated with loan approvals.
- **Reliable Predictions on New Data:** The model generalizes well to new, unseen test data, showcasing its robustness for future deployment or further development in production settings.

## Future Enhancements
Possible improvements and additional steps:

- Experiment with other neural network architectures (e.g., LSTM, GRU).
- Perform feature selection to reduce model complexity and improve interpretability.
- Include cross-validation to better estimate model performance.

