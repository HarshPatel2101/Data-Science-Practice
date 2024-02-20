import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
from sklearn.linear_model import LinearRegression  # Assuming LinearRegression is your model type
import joblib
# Load your machine learning model (corrected)
try:
    model = joblib.load('estate_model.joblib')  # Corrected model loading
except FileNotFoundError:
    raise RuntimeError("Error: 'estate_model.joblib' not found. Please make sure the model file exists.")

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Real Estate Price Prediction", style={'text-align': 'center'}),
        html.Div([
            dcc.Input(id='distance_to_mrt', type='number', placeholder='Distance to MRT Station (meters)',
                      style={'margin': '10px', 'padding': '10px', 'width': '300px'}),
            dcc.Input(id='num_convenience_stores', type='number', placeholder='Number of Convenience Stores',
                      style={'margin': '10px', 'padding': '10px', 'width': '300px'}),
            dcc.Input(id='latitude', type='number', placeholder='Latitude',
                      style={'margin': '10px', 'padding': '10px', 'width': '300px'}),
            dcc.Input(id='longitude', type='number', placeholder='Longitude',
                      style={'margin': '10px', 'padding': '10px', 'width': '300px'}),
            html.Button('Predict Price', id='predict_button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white'}),
        ], style={'text-align': 'center', 'display': 'flex', 'flex-direction': 'column'}),
        html.Div(id='prediction_output', style={'text-align': 'center', 'font-size': '20px', 'margin-top': '20px'}),
    ], style={'width': '50%', 'margin': '0 auto', 'border': '2px solid #007BFF', 'padding': '20px', 'border-radius': '10px'})
])

# Define callback to update output
@app.callback(
    Output('prediction_output', 'children'),
    [Input('predict_button', 'n_clicks')],
    [State('distance_to_mrt', 'value'),
     State('num_convenience_stores', 'value'),
     State('latitude', 'value'),
     State('longitude', 'value')]
)
def update_output(n_clicks, distance_to_mrt, num_convenience_stores, latitude, longitude):
    try:
        if n_clicks > 0:
            # Check if all input values are provided
            if all(v is not None for v in [distance_to_mrt, num_convenience_stores, latitude, longitude]):
                # Prepare the feature vector
                features = pd.DataFrame([[distance_to_mrt, num_convenience_stores, latitude, longitude]],
                                       columns=['distance_to_mrt', 'num_convenience_stores', 'latitude', 'longitude'])

                # Predict
                prediction = model.predict(features)[0]

                return f'Predicted House Price of Unit Area: ${prediction:.2f}'
            else:
                return 'Please enter all values to get a prediction.'
        return ''  # No prediction yet
    except Exception as e:
        return f'Error: {e}'  # Informative error message

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)