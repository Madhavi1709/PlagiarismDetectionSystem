from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Take input from user
print("Enter first text:")
text1 = input()

print("\nEnter second text:")
text2 = input()

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Convert texts into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform([text1, text2])

# Compute cosine similarity
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Display result
score = similarity[0][0]
print("\nCosine Similarity Score: {:.2f}".format(score))

# Interpret score
if score > 0.8:
    print("High similarity — possible plagiarism.")
elif score > 0.5:
    print("Moderate similarity — review recommended.")
else:
    print("Low similarity — texts are likely original.")
