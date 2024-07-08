import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib

st.title("""
Machine Learning using CSV Data
Explore different classifiers and datasets - Which one performs the best?
""")

# Sidebar
st.sidebar.title('Upload CSV file')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV file
    df = pd.read_csv(uploaded_file)

    # Display dataset
    st.write('## Dataset Preview')
    st.write(df.head())

    # Dataset preprocessing
    st.write('## Dataset Information')
    st.write(f'Shape of dataset: {df.shape}')
    st.write(f'Columns: {df.columns.tolist()}')

    # Select features and target
    features = st.sidebar.multiselect("Select Features", df.columns.tolist())
    target = st.sidebar.selectbox("Select Target", df.columns.tolist())

    X = df[features]
    y = df[target]

    # Machine learning model selection
    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('SVM', 'Random Forest', 'Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Linear Regression', 'Neural Network')
    )

    # Function to add classifier parameters
    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
        elif clf_name == 'Random Forest':
            max_depth = st.sidebar.slider('Max Depth', 2, 15)
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            params['max_depth'] = max_depth
            params['n_estimators'] = n_estimators
        elif clf_name == 'Logistic Regression':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
        elif clf_name == 'K-Nearest Neighbors':
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K
        elif clf_name == 'Decision Tree':
            max_depth = st.sidebar.slider('Max Depth', 2, 15)
            params['max_depth'] = max_depth
        return params

    # Function to get classifier
    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'])
        elif clf_name == 'Random Forest':
            clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
        elif clf_name == 'Logistic Regression':
            clf = LogisticRegression(C=params['C'])
        elif clf_name == 'K-Nearest Neighbors':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        elif clf_name == 'Decision Tree':
            clf = DecisionTreeClassifier(max_depth=params['max_depth'])
        elif clf_name == 'Linear Regression':
            clf = LinearRegression()
        elif clf_name == 'Neural Network':
            clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
        return clf

    # Function to train and evaluate model
    def train_model(X_train, y_train, classifier):
        classifier.fit(X_train, y_train)
        return classifier

    # Load a pre-trained model
    def load_model(file_path):
        return joblib.load(file_path)

    # Check if model file is uploaded
    uploaded_model = st.sidebar.file_uploader("Choose a model file (optional)", type="pkl")

    if uploaded_model is not None:
        # Load the model
        clf = load_model(uploaded_model)
        st.sidebar.success('Model loaded successfully')

        # Train and evaluate the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        trained_model = train_model(X_train, y_train, clf)

        if isinstance(clf, (LinearRegression, MLPClassifier)):
            # Regression evaluation
            y_pred = trained_model.predict(X_test)
            if isinstance(clf, LinearRegression):
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f'## {classifier_name} Regression Performance')
                st.write(f'Mean Squared Error: {mse:.2f}')
                st.write(f'R2 Score: {r2:.2f}')
            elif isinstance(clf, MLPClassifier):
                acc = accuracy_score(y_test, y_pred)
                st.write(f'## {classifier_name} Classifier Performance')
                st.write(f'Accuracy: {acc:.2f}')
        else:
            # Classification evaluation
            y_pred = trained_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f'## {classifier_name} Classifier Performance')
            st.write(f'Accuracy: {acc:.2f}')

        # Allow the user to make predictions
        st.sidebar.subheader('Make Predictions')

        # Initialize prediction dictionary
        prediction = {}

        # Display input fields for each selected feature
        for feature in features:
            prediction[feature] = st.sidebar.text_input(f'Enter value for {feature}', '')

        # Create a predict button
        predict_button = st.sidebar.button('Predict')

        # Check if Predict button is clicked and perform prediction
        if predict_button:
            try:
                # Convert input to numerical values
                input_data = np.array([[float(prediction[feature]) for feature in features]])
                result = trained_model.predict(input_data)
                st.success(f'Prediction result: {result}')
            except ValueError:
                st.error('Invalid input. Please enter numerical values.')

    else:
        # Train selected model
        params = add_parameter_ui(classifier_name)
        clf = get_classifier(classifier_name, params)

        if st.sidebar.button('Train Model'):
            # Train and evaluate the model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
            trained_model = train_model(X_train, y_train, clf)

            if isinstance(clf, (LinearRegression, MLPClassifier)):
                # Regression evaluation
                y_pred = trained_model.predict(X_test)
                if isinstance(clf, LinearRegression):
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f'## {classifier_name} Regression Performance')
                    st.write(f'Mean Squared Error: {mse:.2f}')
                    st.write(f'R2 Score: {r2:.2f}')
                elif isinstance(clf, MLPClassifier):
                    acc = accuracy_score(y_test, y_pred)
                    st.write(f'## {classifier_name} Classifier Performance')
                    st.write(f'Accuracy: {acc:.2f}')
            else:
                # Classification evaluation
                y_pred = trained_model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write(f'## {classifier_name} Classifier Performance')
                st.write(f'Accuracy: {acc:.2f}')

            # Allow the user to make predictions
            st.sidebar.subheader('Make Predictions')

            # Initialize prediction dictionary
            prediction = {}

            # Display input fields for each selected feature
            for feature in features:
                prediction[feature] = st.sidebar.text_input(f'Enter value for {feature}', '')

            # Create a predict button
            predict_button = st.sidebar.button('Predict')

            # Check if Predict button is clicked and perform prediction
            if predict_button:
                try:
                    # Convert input to numerical values
                    input_data = np.array([[float(prediction[feature]) for feature in features]])
                    result = trained_model.predict(input_data)
                    st.success(f'Prediction result: {result}')
                except ValueError:
                    st.error('Invalid input. Please enter numerical values.')

else:
    st.warning('Please upload a CSV file to start.')

