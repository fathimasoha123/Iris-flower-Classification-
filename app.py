import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="wide")

st.title("🌸 Iris Flower Classification App")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

st.success(f"Model Accuracy: {accuracy*100:.2f}%")

# Sidebar input
st.sidebar.header("🌿 Enter Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction
if st.button("Predict Species"):

    prediction = model.predict(features)[0]
    species = target_names[prediction]

    st.subheader("Prediction Result")
    st.success(f"🌼 Predicted Species: **{species}**")

    # Probability chart
    probs = model.predict_proba(features)[0]

    prob_df = pd.DataFrame({
        "Species": target_names,
        "Probability": probs
    })

    st.subheader("Prediction Probability")
    st.bar_chart(prob_df.set_index("Species"))

# Dataset preview
st.subheader("📊 Iris Dataset Preview")

df = pd.DataFrame(X, columns=feature_names)
df["species"] = y

st.dataframe(df.head())

# Visualization
st.subheader("📈 Feature Visualization")
st.line_chart(df[["sepal length (cm)", "petal length (cm)"]])

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
        
