
# Week 4. Text as Data Problem Set
# Songtao Duan

#### Conceptual questions ###
# Q1: Bag of words vs. embedding-based representations: 
# Bag-of-words are highly interpretable because each feature corresponds to an actual word count,
# but they are often less robust since they ignore word order and meaning (they prioritize transparency over semantics).
# Embedding representations capture semantic similarity and context, making them more robust for measuring meaning, 
# but they are harder to interpret and can introduce opaque biases that complicate downstream causal or explanatory modeling. 
# For downstream modeling, BoW works well with transparent models like topic models, while embeddings often improve predictive 
# performance in classifiers but reduce explainability. For example, I would prefer BoW when studying framing differences in 
# foreign policy speeches (where specific keywords matter), but I would prefer transformer embeddings when measuring 
# latent ideological similarity or detecting subtle hate speech across different phrasing.


### Applied Exercises ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import hdbscan
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.lda_model
import seaborn as sns
from sklearn.manifold import TSNE

# load the dataset 
df = pd.read_csv("/Users/songtao/Dropbox/26SP/SODA 501/soda501_sp26/04_text_as_data_ps/data_raw/week_movie_corpus.csv")


# LDA: tokenizatu=ion -> dtm -> topics
vectorizer = CountVectorizer(lowercase=True,stop_words="english",token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",min_df=5)

X_counts = vectorizer.fit_transform(df["text"].astype(str))
vocab = vectorizer.get_feature_names_out()

lda = LatentDirichletAllocation(n_components=6,random_state=12345,learning_method="batch")
lda.fit(X_counts)

topic_word = lda.components_
n_top_words = 12

for k in range(6):
    top_word_idx = topic_word[k].argsort()[::-1][:n_top_words]
    words = vocab[top_word_idx]
    print(f"Topic {k}: {', '.join(words)}")

doc_topic_dist = lda.transform(X_counts)

df["dominant_topic"] = doc_topic_dist.argmax(axis=1)

topic_counts = df["dominant_topic"].value_counts().sort_index()

plt.figure(figsize=(9, 5))
plt.bar(topic_counts.index.astype(str), topic_counts.values, color='skyblue', edgecolor='black')
plt.title("Number of Documents per Dominant Topic (LDA)")
plt.xlabel("Topic Label")
plt.ylabel("Document Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("/Users/songtao/Dropbox/26SP/SODA 501/soda501_sp26/04_text_as_data_ps/figures/lda_topic_modeling.png", dpi=200)
plt.show()

### LDA results suggest the corpus is dominated by Topic 2 (around ~260 documents), meaning this theme is the most common storyline pattern in the dataset,
###  while Topics 3–5 are much smaller and likely represent more specialized genres or plot types.


# Word embeddings: Word2Vec -> document vectors -> out of sample prediction
tokenized_docs = [re.findall(r"(?u)\b\w+\b", str(text).lower()) for text in df["text"]]

w2v_model = Word2Vec(sentences=tokenized_docs,vector_size=100,window=5,min_count=2,workers=4,seed=123)

doc_vectors = []
for tokens in tokenized_docs:
    valid_vectors = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if len(valid_vectors) > 0:
        doc_vectors.append(np.mean(valid_vectors, axis=0))
    else:
        doc_vectors.append(np.zeros(w2v_model.vector_size))

X = np.array(doc_vectors)
y = df["y_outcome"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12345)

ridge = Ridge(alpha=1.0, random_state=123)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Fit Line')
plt.title("Word2Vec + Ridge: Actual vs. Predicted")
plt.xlabel("Actual Outcome (Binary: 0 or 1)")
plt.ylabel("Predicted Outcome (Continuous)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("/Users/songtao/Dropbox/26SP/SODA 501/soda501_sp26/04_text_as_data_ps/figures/word2vec_regression_actual_vs_predicted.png", dpi=200)
plt.show()


# BERTopic: transformer embeddings → clustering → topic summaries
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(df["text"].tolist(), show_progress_bar=True)

print("\n--- Transformer embedding matrix ---")
print("Shape:", embeddings.shape)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=123
)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=5,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(df["text"].tolist(), embeddings)

df_bert = df.copy()
df_bert["bertopic_topic"] = topics
df_bert["bertopic_max_prob"] = np.max(probs, axis=1)

print("\n--- BERTopic: topic counts ---")
print(pd.Series(topics).value_counts().sort_index())

topic_info = topic_model.get_topic_info()
print("\n--- BERTopic: topic info (head) ---")
print(topic_info.head(10))

df_bert.to_csv("/Users/songtao/Dropbox/26SP/SODA 501/soda501_sp26/04_text_as_data_ps/data_processed/week_with_bertopic.csv", index=False)
topic_info.to_csv("/Users/songtao/Dropbox/26SP/SODA 501/soda501_sp26/04_text_as_data_ps/outputs/week_bertopic_topic_info.csv", index=False)

topic_counts_bt = topic_info.loc[topic_info["Topic"] != -1, ["Topic", "Count"]]
plt.figure(figsize=(8, 4))
plt.bar(topic_counts_bt["Topic"].astype(str), topic_counts_bt["Count"])
plt.title("BERTopic: Topic Counts (Excluding Outliers)")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("/Users/songtao/Dropbox/26SP/SODA 501/soda501_sp26/04_text_as_data_ps/figures/week_bertopic_topic_counts.png", dpi=200)
plt.show()
plt.close()

 # Regressing an outcome on embeddings means using vector representations of text (or other unstructured data) as predictors in a statistical model, treating each embedding dimension as a feature that captures semantic or contextual information.
 # Wanted as researchers to incorporate complex information from language—such as tone, topic, or meaning—into regression frameworks without manually coding variables.

# A second limitation is domain shift, where embeddings trained on one corpus may not represent meaning accurately in another context, leading to biased estimates. 
