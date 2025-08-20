# Graduate-Admission-Prediction
This project predicts the chances of a student's admission to a graduate school based on various academic factors. An Artificial Neural Network (ANN) is built using TensorFlow and Keras to perform this prediction.

📋 Dataset
The dataset used in this project is the "Graduate Admissions" dataset, which contains the following parameters for 500 students:
GRE Score: Graduate Record Examination score (out of 340).
TOEFL Score: Test of English as a Foreign Language score (out of 120).
University Rating: Rating of the university (out of 5).
SOP: Statement of Purpose strength (out of 5).
LOR: Letter of Recommendation strength (out of 5).
CGPA: Cumulative Grade Point Average (out of 10).
Research: Research experience (0 for no, 1 for yes).
Chance of Admit: The target variable, representing the probability of admission (ranging from 0 to 1).
⚙️ Project Workflow
Data Loading and Preprocessing:
The dataset is loaded using the pandas library.
The 'Serial No.' column is dropped as it is irrelevant for prediction.
The data is checked for any missing values or duplicates.
The features (X) and the target variable (y) are separated.
Data Splitting and Scaling:
The dataset is split into training and testing sets using an 80:20 ratio.
The features are scaled using MinMaxScaler to normalize the data between 0 and 1, which helps in the efficient training of the neural network.
Model Building:
An ANN is constructed using the Keras Sequential API.
The model has the following architecture:
Input Layer: 7 neurons (corresponding to the 7 input features) with a 'relu' activation function.
Hidden Layer: 7 neurons with a 'relu' activation function.
Output Layer: 1 neuron with a 'linear' activation function to predict the continuous value of 'Chance of Admit'.
Model Compilation and Training:
The model is compiled with the Adam optimizer and mean squared error as the loss function.
The model is trained on the scaled training data for 100 epochs with a validation split of 20%.
Model Evaluation:
The trained model is used to make predictions on the scaled test data.
The performance of the model is evaluated using the R-squared (R²) metric, which resulted in a score of approximately 0.73.
The training and validation loss are plotted to visualize the model's learning process.
🛠️ Technologies and Libraries Used
Python 3
NumPy: For numerical operations.
Pandas: For data manipulation and analysis.
Scikit-learn: For data preprocessing and model evaluation (train_test_split, MinMaxScaler, r2_score).
TensorFlow & Keras: For building and training the neural network.
Matplotlib: For plotting the training and validation loss.
🚀 How to Run the Project
Ensure you have Python and the required libraries installed. You can install them using pip:
pip install numpy pandas scikit-learn tensorflow matplotlib


Download the Admission_Predict_Ver1.1.csv dataset.
Place the dataset in the same directory as your code or provide the correct path.
Run the Jupyter Notebook or Python script.

