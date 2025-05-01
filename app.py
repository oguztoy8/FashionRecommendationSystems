import streamlit as st
import numpy as np
import tempfile
from PIL import Image as PILImage  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
import os


# Load model, features, and k-NN index into memory
@st.cache_resource(show_spinner="Loading model and features...")
def load_assets():
    feature_extractor = load_model("resnet50_feature_extractor.keras")
    features = np.load("image_features_resnet50.npy")
    filenames = np.load("filenames.npy", allow_pickle=True)
    knn = NearestNeighbors(
        n_neighbors=6,
        algorithm="brute",
        metric="euclidean"
    )
    knn.fit(features)
    return feature_extractor, features, filenames, knn


# Extract feature vector for a single image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    vec = model.predict(arr, verbose=0).flatten()
    vec /= np.linalg.norm(vec) + 1e-10  # L2-normalize
    return vec


st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="🧥",
    layout="wide"
)

st.title("🧥 Fashion Product Recommendation System")
st.markdown(
    "Upload a product image and the system will recommend **5 similar products** from the collection."
)

uploaded = st.file_uploader(
    "Choose an image (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        query_path = tmp.name

    # Display the uploaded image
    st.subheader("Selected Product")
    st.image(query_path, use_container_width=False)

    # Load model and k-NN
    model, _, filenames, knn = load_assets()

    # Extract features and find nearest neighbors
    query_vec = extract_features(query_path, model)
    dists, idxs = knn.kneighbors([query_vec], n_neighbors=6)

    # Display 5 similar product recommendations in a grid
    st.subheader("Recommended Similar Products")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        img_index = int(idxs[0][i + 1])  
        neighbor_fp = filenames[img_index]
        col.image(neighbor_fp, caption=f"Recommendation {i + 1}", use_container_width=True)

    
    os.remove(query_path)
else:
    st.info("Please upload an image to proceed.")
