import os
import tempfile
from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create the Flask application instance
app = Flask(__name__)

# A simple, single HTML file with Tailwind CSS for styling
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal Data Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        .container {
            max-width: 800px;
        }
    </style>
</head>
<body class="bg-slate-900 text-white min-h-screen flex items-center justify-center p-6">
    <div class="bg-slate-800 p-8 rounded-2xl shadow-2xl w-full container">
        <h1 class="text-3xl font-bold mb-6 text-center text-white">Universal Data Analyzer</h1>
        <p class="text-center text-slate-400 mb-8">
            Upload your dataset (CSV/TXT) and specify the target variable to get a predictive model's performance metrics.
        </p>
        <form action="/analyze" method="post" enctype="multipart/form-data" class="flex flex-col items-center space-y-4">
            <label class="w-full">
                <span class="text-white">Choose file:</span>
                <input type="file" name="file" required
                       class="block w-full text-sm text-slate-500
                       file:mr-4 file:py-2 file:px-4
                       file:rounded-full file:border-0
                       file:text-sm file:font-semibold
                       file:bg-violet-50 file:text-violet-700
                       hover:file:bg-violet-100 cursor-pointer mt-2" />
            </label>
            <label class="w-full">
                <span class="text-white">Target Variable Column Name:</span>
                <input type="text" name="target_column" required
                       class="mt-2 block w-full px-3 py-2 bg-white border border-slate-300 rounded-md text-sm shadow-sm placeholder-slate-400
                       focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 text-slate-900"
                       placeholder="e.g., 'RUL', 'price', 'churn'" />
            </label>
            <button type="submit"
                    class="px-6 py-2 bg-purple-600 text-white font-semibold rounded-full
                           hover:bg-purple-700 transition-colors duration-200">
                Analyze Dataset
            </button>
        </form>

        {% if result %}
        <div class="mt-8 p-6 bg-slate-700 rounded-lg">
            <h2 class="text-xl font-bold mb-4 text-center">Analysis Results</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-center">
                <div class="bg-slate-600 p-4 rounded-lg">
                    <p class="text-sm text-slate-300">Mean Absolute Error (MAE)</p>
                    <p class="text-2xl font-bold text-green-400 mt-1">{{ result.mae }}</p>
                </div>
                <div class="bg-slate-600 p-4 rounded-lg">
                    <p class="text-sm text-slate-300">Root Mean Squared Error (RMSE)</p>
                    <p class="text-2xl font-bold text-green-400 mt-1">{{ result.rmse }}</p>
                </div>
            </div>
            <p class="mt-4 text-sm text-slate-400 text-center">
                The model was trained to predict the selected target variable from the uploaded data.
            </p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# The home page route
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, result=None)

# The file upload and analysis route
@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if a file and target column name were provided
    if 'file' not in request.files or 'target_column' not in request.form:
        return "Missing file or target column name."
    
    file = request.files['file']
    target_column = request.form['target_column']
    
    if file.filename == '':
        return "No selected file"

    if file:
        # Use tempfile to create a secure, cross-platform temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        try:
            # Save the file to the temporary location
            file.save(temp_file.name)

            # --- Universal Data Science Pipeline ---
            # Step 1: Load the dataset (assuming it has a header)
            df = pd.read_csv(temp_file.name, sep=None, engine='python')

            # Step 2: Clean the data
            df.dropna(axis=1, how='all', inplace=True)
            std_dev = df.std(numeric_only=True)
            constant_columns = std_dev[std_dev == 0].index
            df.drop(columns=constant_columns, inplace=True, errors='ignore')

            # Step 3: Define features (X) and target (y)
            if target_column not in df.columns:
                return f"Error: Target column '{target_column}' not found in the dataset."

            y = df[target_column]
            X = df.drop(columns=[target_column])

            # Filter for numerical features only
            numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
            X = X[numeric_cols]
            
            if X.empty or len(X) < 2:
                raise ValueError("Dataset is too small or does not contain enough numerical features for analysis.")

            # Step 4: Normalize the features
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Step 5: Split the data for modeling
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Step 6: Build and train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Step 7: Evaluate the model's performance
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Return the results to the user
            result = {
                'mae': f'{mae:.2f}',
                'rmse': f'{rmse:.2f}'
            }
            return render_template_string(HTML_TEMPLATE, result=result)

        except Exception as e:
            # Return a user-friendly error message if something goes wrong
            return f"An error occurred: {e}"

        finally:
            # Ensure the temporary file is deleted even if an error occurs
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

if __name__ == '__main__':
    app.run(debug=True)

