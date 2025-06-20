from sklearn.metrics.pairwise import cosine_similarity

def get_top_recommendations(tfidf_matrix, user_vector, top_n=5):
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    top_indices = similarities.argsort()[0][::-1][:top_n]
    return top_indices, similarities[0][top_indices]
