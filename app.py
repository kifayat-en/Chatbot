from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__, template_folder="templates")

lemmatizer = WordNetLemmatizer()

# Load chatbot data
intents = json.loads(open("intents.json").read())
words = pickle.load(open("models/words.pkl", "rb"))
classes = pickle.load(open("models/classes.pkl", "rb"))
model = load_model("models/chatbot_model.h5")

# Preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_vector = bow(sentence, words, show_details=False)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "Sorry, I didn't understand that. Can you rephrase?"

    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i["responses"])

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    user_message = data.get("message")

    intents_list = predict_class(user_message)
    response = get_response(intents_list, intents)

    return jsonify({"response": response})

if __name__ == "__main__":
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    app.run(host="0.0.0.0", port=5000, debug=True)

