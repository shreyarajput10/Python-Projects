#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:50:35 2024

@author: HP
"""
import pandas as pd
import spacy
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline

# Load spaCy's English NLP model
nlp = spacy.load("en_core_web_sm")

# Load the tokenizer and model from the pre-trained Roberta sentiment analysis model
model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Create a sentiment analysis pipeline
sentiment_analysis_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Label Mapping (if needed, based on the output from your model)
label_map = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}

# Function to perform sentiment analysis
def sentiment_analysis(text):
    results = sentiment_analysis_pipeline(text)
    # Map the model's label to a more readable form
    readable_label = label_map.get(results[0]['label'], results[0]['label'])
    score = results[0]['score']
    return readable_label, score

# Load your dataset
data_path = '/Users/HP/Desktop/Exercise1/Houseofdragons/HouseOfDragons.csv'  # Replace with your actual path
data = pd.read_csv(data_path)

# Function to preprocess text and remove stop words
def preprocess_text(text):
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    text = text.lower().strip()
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    return text

# Apply preprocessing to the MsgBody column
data['Processed_MsgBody'] = data['MsgBody'].apply(preprocess_text)

# Function to perform NER and extract entities
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text.strip(), ent.label_) for ent in doc.ents]

# Apply NER to the preprocessed MsgBody column
data['Entities'] = data['Processed_MsgBody'].apply(extract_entities)

# Create a list of known characters (extend this list based on your knowledge)
known_characters = ['robert']

# Filter entities to keep only known characters
data['Character_Entities'] = data['Entities'].apply(
    lambda entities: [entity for entity in entities if entity[0].lower() in known_characters]
)

# Extract sentences with characters
data['Sentences_with_Characters'] = data.apply(
    lambda row: [sent.text.strip() for ent in row['Character_Entities'] for sent in nlp(row['Processed_MsgBody']).sents if ent[0] in sent.text],
    axis=1
)

# Flatten the list of sentences for sentiment analysis
sentences_for_sentiment_analysis = []
for index, row in data.iterrows():
    for sentence in row['Sentences_with_Characters']:
        sentences_for_sentiment_analysis.append(sentence)

# Convert the list to a DataFrame for sentiment analysis
sentences_df = pd.DataFrame(sentences_for_sentiment_analysis, columns=['Sentence'])

# Perform sentiment analysis
sentences_df['Sentiment'], sentences_df['Score'] = zip(*sentences_df['Sentence'].apply(sentiment_analysis))

# Filter out Neutral comments
sentences_df = sentences_df[sentences_df['Sentiment'] != 'Neutral']

# Calculate sentiment counts and average scores
summary_df = sentences_df.groupby('Sentiment').agg(
    #Count=('Sentiment', 'count'),
    Average_Score=('Score', 'mean')
).reset_index()

# Save the detailed DataFrame and summary DataFrame to CSV files
sentences_df.to_csv('/Users/HP/Desktop/Exercise1/Houseofdragons/HOD(robert).csv', index=False)
summary_df.to_csv('/Users/HP/Desktop/Exercise1/Houseofdragons/Sentiment Summary Robert.csv', index=False)

# Display the sentiments
print(sentences_df)
print(summary_df)






