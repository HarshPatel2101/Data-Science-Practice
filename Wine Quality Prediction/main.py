import pickle
from flask import Flask, render_template, request
import os  # For error handling

# Error handling for model loading
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print(f"Error loading model: Model file 'model.pkl' not found. Ensure it exists at path: {os.path.abspath('model.pkl')}")
    quit()  # Terminate gracefully

app = Flask(__name__)

# Route for form submission and prediction
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Create a dictionary to store form inputs
        form_inputs = {}
        for field in ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'Residual_sugar', 'chlorides',
                      'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']:
            if field in request.form:
                # Validate input type:
                try:
                    form_inputs[field] = float(request.form[field])  # Assuming numerical features
                except ValueError:
                    print(f"Error: Invalid input for '{field}'. Expected a number.")
                    return render_template('index.html', error=f"Invalid input for '{field}'.")
            else:
                print(f"Warning: Missing input for '{field}'.")

        # Make the prediction using the model
        # Make the prediction using the model
            try:
                prediction = model.predict([list(form_inputs.values())])[0]
            except Exception as e:
                print(f"Error during prediction: {e}")
                prediction = "An error occurred."

        # Render the template with prediction and input values
        return render_template('index.html', prediction=prediction, form_inputs=form_inputs)
    else:
        # Render the initial form template
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
