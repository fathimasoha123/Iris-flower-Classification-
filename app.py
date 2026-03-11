import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

st.title("🌸 Iris Flower Classification App")
st.write("Predict the Iris species using machine learning.")

# Load model
try:
    model = joblib.load("iris_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar inputs
st.sidebar.header("🌿 Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Feature array
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction
if st.button("🔍 Predict Species"):

    prediction = model.predict(features)
    species = label_encoder.inverse_transform(prediction)

    st.success(f"🌼 Predicted Species: **{species[0]}**")

    # Probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]

        prob_df = pd.DataFrame({
            "Species": label_encoder.classes_,
            "Probability": probs
        })

        st.subheader("Prediction Probability")
        st.bar_chart(prob_df.set_index("Species"))

# Show feature values
st.subheader("Input Feature Values")

feature_df = pd.DataFrame({
    "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
    "Value": [sepal_length, sepal_width, petal_length, petal_width]
})

st.table(feature_df)

# Dataset preview
st.subheader("📊 Iris Dataset Preview")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

st.dataframe(df.head())

# Visualization
st.subheader("📈 Feature Visualization")

chart_data = df[["sepal length (cm)", "petal length (cm)"]]

st.line_chart(chart_data)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")