import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Collaborative Filtering Recommendation
def get_cf_recommendations(user_id, model, ratings_df, top_n=5, data=None):
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    if user_ratings.empty:
        return []

    all_products = ratings_df['parent_asin'].unique()
    user_products = user_ratings['parent_asin'].tolist()
    not_rated = [pid for pid in all_products if pid not in user_products]

    predictions = []
    for pid in not_rated:
        try:
            pred = model.predict(user_id, pid)
            predictions.append((pid, pred.est))
        except:
            continue  

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_pids = [pid for pid, _ in predictions[:top_n]]

    if data is not None:
        return [get_product_info(pid, data) for pid in top_pids]

    return top_pids


# Content-Based Recommendation
def get_similar_products_by_title(title, df, tfidf_matrix, faiss_index, top_n=5):
    idx = df[df['title_y'] == title].index
    if len(idx) == 0:
        return f"'{title}' not found in the dataset."

    idx = idx[0]

    print(f"[DEBUG] idx = {idx}")
    print(f"[DEBUG] tfidf_matrix type: {type(tfidf_matrix)}")
    print(f"[DEBUG] tfidf_matrix[idx] = {tfidf_matrix[idx]}")

    try:
        query_vector = tfidf_matrix[idx].toarray().astype('float32')
    except Exception as e:
        return f"[ERROR] Failed to extract vector: {e}"

    D, I = faiss_index.search(query_vector, top_n + 1)
    similar_indices = [i for i in I[0] if i != idx][:top_n]

    return df[['title_y', 'price']].iloc[similar_indices].reset_index(drop=True).to_dict(orient="records")


# Hybrid Recommendation using FAISS + Collaborative
def hybrid_recommend(user_id, model, faiss_index, product_ids, ratings_df, top_n=5, data=None):
    cf_recs = get_cf_recommendations(user_id, model, ratings_df, top_n=top_n * 2)
    hybrid_scores = []

    for pid in cf_recs:
        if pid not in product_ids:
            continue

        try:
            idx = product_ids.index(pid)
        except ValueError:
            continue 

        try:
            query_vector = faiss_index.reconstruct(idx)
        except:
            continue

        scores, indices = faiss_index.search(np.array([query_vector]), 2)

        if indices[0][0] == idx:
            similar_idx = indices[0][1]
        else:
            similar_idx = indices[0][0]

        if 0 <= similar_idx < len(product_ids):
            similar_pid = product_ids[similar_idx]
            hybrid_scores.append(similar_pid)

    unique_scores = list(dict.fromkeys(hybrid_scores))[:top_n]

    if data is not None:
        return [get_product_info(pid, data) for pid in unique_scores]

    return unique_scores



def get_product_info(parent_asin, data):
    product_row = data[data['parent_asin'] == parent_asin]

    if product_row.empty:
        return {'title_y': 'Unknown', 'price': 'N/A'}
    
    title = product_row.iloc[0]['title_y']
    price = product_row.iloc[0]['price']
    
    return {'title_y': title, 'price': price}
