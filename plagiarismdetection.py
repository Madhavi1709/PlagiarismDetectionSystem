from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example texts (you can replace with your own)
text1 = """Machine learning is a field of artificial intelligence that focuses 
on enabling computers to learn from data and improve over time without being explicitly programmed."""

text2 = """Artificial intelligence includes machine learning, where computers learn 
from data and get better over time without specific programming."""

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Convert texts into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform([text1, text2])

# Compute cosine similarity
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Display result
print("Cosine Similarity Score: {:.2f}".format(similarity[0][0]))

# Optional: interpret score
if similarity[0][0] > 0.8:
    print("  High similarity — possible plagiarism.")
elif similarity[0][0] > 0.5:
    print(" Moderate similarity — review recommended.")
else:
    print(" Low similarity — texts are likely original.")
