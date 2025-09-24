PROJECT TITLE : UNIVERSAL DATA ANALYSER

This is a web application that provides a universal tool for predictive modeling. Users can upload any structured dataset (in CSV or TXT format) and get key performance metrics for a machine learning model trained on their data.

The application automatically handles the entire data science pipeline, from data cleaning to model evaluation, making it a plug-and-play solution for quick data analysis.

Features and MethodS
File Uploads: Accepts any structured dataset in a .csv or .txt format.

Dynamic Analysis: The user provides the name of the target variable (the column they want to predict).

Automatic Data Cleaning: Automatically handles missing and constant columns to prepare the data for modeling.

Model Training: Trains a Random Forest Regressor, a powerful and versatile machine learning model, on the uploaded dataset.

Performance Metrics: Returns two key metrics to evaluate the model's accuracy:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Technologies Used 
Backend: Python, Flask

Data Science: Pandas, NumPy, Scikit-learn

Frontend: HTML5, Tailwind CSS

Steps to run:
Prerequisites: Ensure you have Python installed.

Install Libraries: Open your terminal and install the required libraries:

pip install Flask pandas numpy scikit-learn

Run the Server: Navigate to the project directory in your terminal and run the app.py file:

python app.py

Access the App: Open your web browser and go to http://127.0.0.1:5000 to start analyzing your datasets.
