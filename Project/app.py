from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained logistic regression model
model = pickle.load(open('logistic_regression_model_balanced.pkl', 'rb'))  # Replace with your model file path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the features from the form
        transaction_amount = float(request.form['TransactionAmount'])
        anomaly_score = float(request.form['AnomalyScore'])
        amount = float(request.form['Amount'])
        account_balance = float(request.form['AccountBalance'])
        suspicious_flag = int(request.form['SuspiciousFlag'])
        gap = float(request.form['gap'])
        
        # Prepare the input data for prediction
        input_data = np.array([transaction_amount, anomaly_score, amount, account_balance, suspicious_flag, gap]).reshape(1, -1)
        
        # Get the prediction from the model
        prediction = model.predict(input_data)
        prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of fraud
        
        # Return the result
        if prediction[0] == 1:
            result = f"Fraudulent transaction detected! (Probability: {prediction_prob*100:.2f}%)"
        else:
            result = f"Transaction is legitimate. (Probability: {(1 - prediction_prob)*100:.2f}%)"
        
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
