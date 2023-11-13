import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Define data variable in the global scope
data = pd.DataFrame()

def get_sentence_embedding(sentence, word2vec_model):
    words = re.findall(r'\w+', sentence.lower())
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return sum(vectors) / len(vectors) if vectors else [0] * 100

def train_genre_classifier(train_data_path, test_data_path):
    global data  # Reference the global 'data' variable
    train_data = pd.read_csv(train_data_path, sep=':::', names=['Title', 'Description', 'Genre'], engine='python')
    test_data = pd.read_csv(test_data_path, sep=':::', names=['Title', 'Description'], engine='python')
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    data = combined_data
    data = data.dropna()
    corpus = data["Description"].apply(lambda x: re.findall(r'\w+', x.lower()))
    word2vec_model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)

    data["embeddings"] = data["Description"].apply(lambda x: get_sentence_embedding(x, word2vec_model))
    X = pd.DataFrame(data["embeddings"].to_list())
    label_encoder = LabelEncoder()
    data["genre_encoded"] = label_encoder.fit_transform(data["Genre"])
    y = data["genre_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, label_encoder, X_test, y_test, word2vec_model

def predict_genre(model, label_encoder, new_descriptions, word2vec_model):
    new_embeddings = [get_sentence_embedding(desc, word2vec_model) for desc in new_descriptions]
    new_X = pd.DataFrame(new_embeddings)
    new_predictions = model.predict(new_X)
    predicted_genres = label_encoder.inverse_transform(new_predictions)

    return predicted_genres

train_data_path = "D:/Internships/codesoft/task1/Genre Classification Dataset/train_data.txt"
test_data_path = "D:/Internships/codesoft/task1/Genre Classification Dataset/test_data.txt"

trained_model, genre_label_encoder, X_test, y_test, word2vec_model = train_genre_classifier(train_data_path, test_data_path)

new_descriptions = ["A group of friends embark on an adventurous journey.", "In a dystopian future, a hero rises to save the world.", "A heartwarming story of family and friendship."]

predicted_genres = predict_genre(trained_model, genre_label_encoder, new_descriptions, word2vec_model)

for desc, genre in zip(new_descriptions, predicted_genres):
    print("Description:", desc)
    print("Predicted Genre:", genre)
    print()

# Evaluation Metrics
predictions = trained_model.predict(X_test)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')

print("Precision:", precision)
print("Recall:", recall)

# Plot Genre Distribution
plt.figure(figsize=(10, 6))
genre_counts = data["Genre"].value_counts()
genre_counts.plot(kind="bar", color=["#FF7F0E", "#1F77B4", "#2CA02C", "#D62728", "#9467BD"])
plt.xlabel("Genre")
plt.ylabel("Count")
plt.title("Genre Distribution")
plt.xticks(rotation=45, ha='right')
plt.show()
