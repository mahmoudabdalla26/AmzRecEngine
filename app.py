from flask import Flask, render_template, request
import pickle
import pandas as pd
import faiss
import numpy as np
import scipy.sparse
from utils.recommenders import get_cf_recommendations, get_similar_products_by_title, hybrid_recommend, get_product_info 

app = Flask(__name__)

# Load preprocessed models and data
collaborative_model = pickle.load(open("models/collaborative_model.pkl", "rb"))
tfidf_matrix = scipy.sparse.load_npz("models/tfidf_matrix.npz")
tfidf_vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
faiss_index = faiss.read_index("models/faiss_index.index")
# product_ids = pickle.load(open("models/product_ids.pkl", "rb"))
product_ids = pickle.load(open("models/product_ids.pkl", "rb"))  
ratings_df = pd.read_csv("models/ratings_subset.csv")
data = pd.read_csv("models/preprocessed_data_sample.csv")
products_df = pd.read_csv("data/preprocessed_data.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    error_message = ""

    product_ids = pickle.load(open("models/product_ids.pkl", "rb"))

    if request.method == "POST":
        user_id = request.form["user_id"]
        method = request.form["method"]

        try:
            if method == "collaborative":
                product_ids = get_cf_recommendations(
                    user_id, collaborative_model, ratings_df, top_n=5
                )
                # Apply get_product_info only for collaborative
                recommendations = [get_product_info(pid, data) for pid in product_ids]

            elif method == "content":
                # content already returns title_y + price directly
                recommendations = get_similar_products_by_title(
                    user_id, data, tfidf_matrix, faiss_index
                )

            elif method == "hybrid":
                product_id = hybrid_recommend(
                        user_id, collaborative_model, faiss_index, product_ids, ratings_df, top_n=5
                    )
                # Apply get_product_info only for hybrid
                recommendations = [get_product_info(pid, data) for pid in product_id]

            else:
                error_message = "Invalid method selected."

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render_template("index.html", recommendations=recommendations, error=error_message)



if __name__ == "__main__":
    app.run(debug=True)