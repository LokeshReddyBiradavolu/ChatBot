pip install nltk scikit-learn // Run this In CMD
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample conversation data
conversations = [
    ("Hi there!", "Hello! How can I help you?"),
    ("How are you?", "I'm doing well, thank you."),
    ("who are you?", " I'm Chat Bot."),
    ("What is your name?", "I'm a chatbot."),
    ("Goodbye", "Goodbye! Have a nice day.")
]

# Tokenization and TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([conv[0] for conv in conversations])

def get_response(user_input):
    # Transform user input to TF-IDF vector
    user_input_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Calculate cosine similarity between user input and conversation responses
    similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)
    
    # Get the index of the most similar response
    most_similar_index = np.argmax(similarities)
    
    # Return the corresponding response
    return conversations[most_similar_index][1]

print("Chatbot: Hello! How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye! Have a nice day.")
        break
    response = get_response(user_input)
    print("Chatbot:", response)
