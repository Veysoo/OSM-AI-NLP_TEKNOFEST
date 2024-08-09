import pathlib
import numpy as np
from flair.models import SequenceTagger
from flair.data import Sentence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import re
import json
import pickle
from tensorflow.keras.models import load_model
import uvicorn

# PosixPath'ı WindowsPath'e yönlendirme
pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

# Model ve tokenizer yükleme
model_path = "C:\\Users\\Lenovo\\Desktop\\nlp_teknofest\\best_model.keras"
tokenizer_path = "C:\\Users\\Lenovo\\Desktop\\nlp_teknofest\\tokenizer.json"
label_encoder_path = "C:\\Users\\Lenovo\\Desktop\\nlp_teknofest\\label_encoder.pkl"
tagger_path = r"C:\Users\Lenovo\Desktop\nlp_teknofest\taggers\ner-finetuned\best-model.pt"

# Load the tokenizer from JSON file
with open(tokenizer_path) as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

# Load the label encoder
with open(label_encoder_path, 'rb') as file:
    le = pickle.load(file)

# Load the sentiment analysis model
model = load_model(model_path)

# Load the NER model once
tagger = SequenceTagger.load(tagger_path)

# Define maximum length
max_length = 200

# Define request model
class Item(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz.  Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell """)

# Entity extraction function
def extract_entities(texts):
    try:
        sentence = Sentence(texts)
        tagger.predict(sentence)
        entities = [entity.text for entity in sentence.get_spans('ner')]
        return entities
    except Exception as e:
        print(f"Error in extract_entities: {e}")
        return []

# Text cleaning function
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+|@\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    #text = text.lower()
    return text

# Get entity sentiments function
def get_entity_sentiments(entities, tokenizer, model, max_length, le):
    if not entities:
        return []

    try:
        sequences = tokenizer.texts_to_sequences(entities)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        predictions = model.predict(padded_sequences)
        predicted_classes = np.argmax(predictions, axis=-1)
        sentiments = le.inverse_transform(predicted_classes)
        return sentiments
    except Exception as e:
        print(f"Error during prediction: {e}")
        return ["Error"] * len(entities)

# Text processing endpoint
@app.post("/predict/")
async def predict(item: Item):
    try:
        cleaned_text = clean_text(item.text)
        entities = extract_entities(cleaned_text)
        sentiments = get_entity_sentiments(entities, tokenizer, model, max_length, le)
        result = {
            "entity_list": entities,
            "results": [{"entity": entity, "sentiment": sentiment} for entity, sentiment in zip(entities, sentiments)]
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
