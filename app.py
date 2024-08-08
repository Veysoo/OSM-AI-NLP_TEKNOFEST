import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from flair.data import Sentence
from flair.models import SequenceTagger
from flask import Flask, request, jsonify
from keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords

# Gerekli dosyayı indir
nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))


# Load pre-existing embedding matrix
embedding_matrix_path = "C:/Users/tayip/Desktop/nlp_proje/embedding_matrix1.npy"
embedding_matrix = np.load(embedding_matrix_path)
print(f"Embedding matrix loaded from {embedding_matrix_path}")

# Define parameters
embedding_dim = embedding_matrix.shape[1]
vocab_size = embedding_matrix.shape[0]
max_length = 200
num_classes = 3  # Adjust based on your number of classes

# Load and prepare the label encoder
data = pd.read_csv("C:/Users/tayip/Desktop/nlp_proje/sentimentdata.csv")
print(data.columns)  # Print column names to verify
all_sentiments = [sent for sublist in data['Sentiments'].apply(eval).values for sent in sublist]
le = LabelEncoder()
le.fit(all_sentiments)

# Initialize and fit Tokenizer on your data
texts = data['Sentiments'].tolist()  
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Load the model weights
model = load_model("C:/Users/tayip/Desktop/nlp_proje/best_model.keras")
print(f"Model weights loaded from best_model.keras")

# Load Flair NER model
nlp_flair = SequenceTagger.load("flair/ner-multi")

# Extract entities using Flair NER
def extract_entities(text):
    sentence = Sentence(text)
    nlp_flair.predict(sentence)
    entities = [entity.text for entity in sentence.get_spans('ner')]
    return entities


# Get sentiments for entities
def get_entity_sentiments(entities, tokenizer, model, max_length, le):
    tokenized_entities = [simple_preprocess(entity) for entity in entities]
    sequences = tokenizer.texts_to_sequences(tokenized_entities)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    predictions = model.predict(padded_sequences)
    predicted_classes = np.argmax(predictions, axis=-1)
    sentiments = le.inverse_transform(predicted_classes)
    return sentiments

# Initialize Flask application
app = Flask(__name__)

# Define endpoint for processing text
@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    text = data['text']
    entities = extract_entities(text)
    sentiments = get_entity_sentiments(entities, tokenizer, model, max_length, le)
    result = {
        "entity_list": entities,
        "results": [{"entity": entity, "sentiment": sentiment} for entity, sentiment in zip(entities, sentiments)]
    }
    return jsonify(result)

# Start Flask application
if __name__ == '__main__':
    app.run(debug=True)
