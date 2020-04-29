from flask import Flask, request, render_template
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from utils import word_feats, get_filtered_words, get_classifier
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        review = request.form.get("review")
        if not review:
            return render_template("error.html"), 404

        words = [word for sent in sent_tokenize(
            review) for word in word_tokenize(sent)]
        
        testfeats = word_feats(get_filtered_words(words, test=True))

        classifier = get_classifier("my_classifier.pickle")
        
        result = classifier.classify(testfeats)
        data = {"result": result, "review": review}
        return render_template("index.html", **data)


    return render_template("index.html")

if __name__ == "__main__":
    app.run()
