import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
# --- NLTK Data Check & Download ---
required_resources = [
    ("stopwords", "corpora/stopwords"),
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),  # important for word_tokenize
    ("vader_lexicon", "sentiment/vader_lexicon"),
    ("wordnet", "corpora/wordnet"),
]

for resource_name, resource_path in required_resources:
    try:
        nltk.data.find(resource_path)
        print(f"[OK] {resource_name} is already installed.")
    except LookupError:
        print(f"[Downloading] {resource_name}...")
        nltk.download(resource_name)



# Load stopwords once
stop_words = set(stopwords.words('english'))

# --- Text Preprocessing ---
def preprocess_text(text):
    """
    Cleans and preprocesses the text data for analysis.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text)  # remove URLs, mentions, hashtags
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return lemmas

def load_and_clean_data(file_path):
    """Loads the dataset and performs initial data cleaning."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    
    df['text'] = df['text'].fillna('')
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Remove empty processed texts
    df = df[df['processed_text'].map(len) > 0]

    return df

# --- Sentiment Analysis ---
def perform_sentiment_analysis(df):
    sid = SentimentIntensityAnalyzer()
    
    def get_vader_sentiment(tokens):
        if not tokens:
            return 'Neutral'
        scores = sid.polarity_scores(" ".join(tokens))
        if scores['compound'] >= 0.05:
            return 'Positive'
        elif scores['compound'] <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    print("Performing sentiment analysis...")
    df['sentiment'] = df['processed_text'].apply(get_vader_sentiment)
    
    # Plot sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['#4CAF50', '#F44336', '#FFEB3B'])
    plt.title('Sentiment Distribution of Comments')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show(block=False)  # non-blocking

    return df

# --- Topic Modeling ---
def perform_topic_modeling(df, num_topics=5, passes=10):
    print("Performing topic modeling...")
    dictionary = Dictionary(df['processed_text'])
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    corpus = [dictionary.doc2bow(text) for text in df['processed_text']]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )

    print("\n--- Discovered Topics ---")
    for topic in lda_model.print_topics(num_words=10):
        print(topic)

    print("\n--- Word Clouds for Topics ---")
    for i, topic_words in lda_model.show_topics(formatted=False):
        words = " ".join([w for w, _ in topic_words])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {i+1} Word Cloud')
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)

# --- Main Execution ---
if __name__ == "__main__":
    file_path = 'climate_nasa.csv'
    data = load_and_clean_data(file_path)
    
    if data is not None:
        data = perform_sentiment_analysis(data)
        perform_topic_modeling(data)

        print("\n--- Sample of Final DataFrame ---")
        print(data[['text', 'sentiment']].head())
