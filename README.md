# AmzRecEngine
# 🧠 HybridRec-System

A **hybrid recommendation system** for Amazon Software products that combines **Collaborative Filtering**, **Content-Based Filtering**, and **Vector Similarity Search (FAISS)** to deliver accurate and personalized product suggestions.

---

## 📦 Project Overview

This project is built to simulate a real-world recommendation engine using data from the [Amazon Reviews 2023 Project](https://amazon-reviews-2023.github.io/).  
It includes a full pipeline from data cleaning and modeling to deploying a web application using Flask.

---

## 🎯 Recommendation Approaches

- **Collaborative Filtering**  
  Suggests items based on user–item interaction matrix using **SVD** algorithm (Surprise library).

- **Content-Based Filtering**  
  Uses **TF-IDF** vectorization of product titles and **FAISS** for similarity search.

- **Hybrid Method**  
  Combines collaborative recommendations and refines them using content-based similarity.

---

## 🧰 Tech Stack

- **Python** (Pandas, NumPy, scikit-learn)
- **Surprise** (SVD for collaborative filtering)
- **FAISS** (Fast Approximate Nearest Neighbor Search)
- **Flask** (Web deployment)
- **Jupyter Notebooks** (EDA, modeling, prototyping)

---

## 📁 Project Structure

```
HybridRec-System/
│
├── app.py ← Main Flask app
│
├── data/ ← Raw Amazon review & metadata
│ ├── Software.jsonl.gz
│ └── meta_Software.jsonl.gz
│
├── models/ ← Trained models and preprocessed data
│ ├── collaborative_model.pkl
│ ├── tfidf_vectorizer.pkl
│ ├── tfidf_matrix.npz
│ ├── faiss_index.index
│ ├── product_ids.pkl
│ └── ratings_subset.csv
│
├── Notebooks/ ← EDA and model notebooks
│ ├── Data_Cleaning_and_EDA.ipynb
│ ├── Collabrative_Filtring.ipynb
│ ├── Content_Based_Recommender.ipynb
│ └── Hybrid_Recommender.ipynb
│
├── templates/
│ └── index.html ← Web UI template
│
├── utils/
│ └── recommenders.py ← All recommendation logic
``` 

## 🚀 How to Run the Project

### 1. Clone the repository
bash
git clone https://github.com/your-username/HybridRec-System.git
cd HybridRec-System

### 2. Install dependencies
Make sure you’re using Python 3.8+

3. Run the Flask app
python app.py

4. Open in browser
Go to: http://127.0.0.1:5000

🖥️ Application Demo
https://drive.google.com/file/d/16ks_soYpGe_yjW9e6ZFXSJj2vg0Q21xP/view?usp=sharing

📊 Data Source
Data used from the Amazon Reviews 2023 project:
Over 100,000 software reviews
Product metadata includes titles, prices, and more

✨ Sample Use Cases

| Method        | Input         | Output                          |
| ------------- | ------------- | ------------------------------- |
| Collaborative | User ID       | Products other users liked      |
| Content-Based | Product Title | Similar products by content     |
| Hybrid        | User ID       | Personalized + similar products |


🧪 Notebooks
Check the /Notebooks folder to explore:
Data cleaning steps
TF-IDF + FAISS modeling
SVD-based collaborative filtering
Hybrid recommender combination logic
