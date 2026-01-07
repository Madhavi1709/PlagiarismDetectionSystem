from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    similarity_score = None
    result = None

    if request.method == "POST":
        text1 = request.form.get("text1", "")
        text2 = request.form.get("text2", "")

        if text1 and text2:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            score = similarity[0][0]

            similarity_score = round(score * 100, 2)

            if score > 0.8:
                result = "High similarity — Possible plagiarism"
            elif score > 0.5:
                result = "Moderate similarity — Review recommended"
            else:
                result = "Low similarity — Texts are likely original"

    return render_template(
        "index.html",
        similarity_score=similarity_score,
        result=result
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
