import os
import json
import numpy as np
import tensorflow as tf
import pickle
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import pandas as pd
import logging

# Configure logging to show INFO and above messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Enable CORS for all endpoints

# --- Model Path Configuration ---
BASE_DIR = os.path.dirname(__file__)

# Sentiment Models Paths
SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'sentiment_model_lstm.h5')
SENTIMENT_TOKENIZER_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'tokenizer.pkl')
LEXICON_POSITIVE_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'lexicon_positive.json')
LEXICON_NEGATIVE_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'lexicon_negative.json')
SLANGWORDS_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'combined_slang_words.txt')
STOPWORDS_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'combined_stop_words.txt')

# Search/Content-Based Filtering Models Paths
CBR_DATA_PATH = os.path.join(BASE_DIR, 'Models', 'search', 'cbr_clean.csv')

# Hybrid Recommender (TFLite) Models Paths
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'recommend', 'model.tflite')
PLACE_ENCODER_PATH = os.path.join(BASE_DIR, 'Models', 'recommend', 'place_encoder.pkl')
USER_ENCODER_PATH = os.path.join(BASE_DIR, 'Models', 'recommend', 'user_encoder.pkl')

# --- Global Variables for ML Assets ---
sentiment_model = None
sentiment_tokenizer = None
lexicon_positive = {}
lexicon_negative = {}
stemmer = None
stopword_remover = None
slangwords = {}
all_stopwords = set()

# For Search/Content-Based Filtering
tfidf_vectorizer = None
tfidf_matrix = None
beach_data_for_search = None

# For Hybrid Recommender
hybrid_recommender_interpreter = None
place_encoder = None
user_encoder = None

MAX_SEQUENCE_LENGTH = 10

# --- Function to Load ML Assets ---
def load_ml_assets():
    """Loads all ML models and other assets when the application starts."""
    global sentiment_model, sentiment_tokenizer, lexicon_positive, lexicon_negative, stemmer, \
           hybrid_recommender_interpreter, place_encoder, user_encoder, stopword_remover

    logging.info("Starting to load ML assets...")
    try:
        # Load Sentiment Models
        logging.info(f"Loading sentiment model from: {SENTIMENT_MODEL_PATH}")
        sentiment_model = tf.keras.models.load_model(SENTIMENT_MODEL_PATH)
        logging.info("Sentiment model loaded successfully.")

        logging.info(f"Loading sentiment tokenizer from: {SENTIMENT_TOKENIZER_PATH}")
        with open(SENTIMENT_TOKENIZER_PATH, 'rb') as f:
            sentiment_tokenizer = pickle.load(f)
        logging.info("Sentiment Tokenizer loaded successfully.")

        # Load Sastrawi
        logging.info("Loading Sastrawi Stemmer and StopWordRemover...")
        stemmer_factory = StemmerFactory()
        stemmer = stemmer_factory.create_stemmer()
        stopword_remover_factory = StopWordRemoverFactory()
        stopword_remover = stopword_remover_factory.create_stop_word_remover()
        logging.info("Sastrawi assets loaded.")
        
        # Load Hybrid Recommender TFLite Model
        logging.info(f"Loading Hybrid Recommender Model from: {TFLITE_MODEL_PATH}")
        hybrid_recommender_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        hybrid_recommender_interpreter.allocate_tensors()
        logging.info("Hybrid Recommender Model (TFLite) loaded successfully.")

        # Load Encoders
        logging.info(f"Loading Place Encoder from: {PLACE_ENCODER_PATH}")
        with open(PLACE_ENCODER_PATH, 'rb') as f:
            place_encoder = pickle.load(f)
        logging.info("Place Encoder loaded successfully.")

        logging.info(f"Loading User Encoder from: {USER_ENCODER_PATH}")
        with open(USER_ENCODER_PATH, 'rb') as f:
            user_encoder = pickle.load(f)
        logging.info("User Encoder loaded successfully.")

    except Exception as e:
        logging.critical(f"FATAL ERROR loading ML assets: {e}", exc_info=True)
        # Set all models to None to prevent the app from running in a broken state
        sentiment_model = None
        hybrid_recommender_interpreter = None
        # ... etc.

# --- Preprocessing Functions ---
def load_slangwords(file_path):
    slangwords_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if ':' in line:
                    key, value = line.split(':', 1)
                    slangwords_dict[key.strip()] = value.strip()
        logging.info(f"Slangwords loaded from: {file_path}")
    except Exception as e:
        logging.error(f"Error loading slangwords from {file_path}: {e}")
    return slangwords_dict

def load_stopwords(file_path):
    stopwords_set = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stopwords_set.add(line.strip())
        logging.info(f"Stopwords loaded from: {file_path}")
    except Exception as e:
        logging.error(f"Error loading stopwords from {file_path}: {e}")
    return stopwords_set

def cleaning_text(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower() # Case folding
    text = text.replace(r'@[\w\d]+', ' ').replace(r'#[\w\d]+', ' ').replace(r'RT[\s]', ' ')
    text = text.replace(r'http\S+|www\S+', ' ').replace(r'\d+', ' ').replace(r'[^a-zA-Z\s]', ' ')
    text = text.replace('\n', ' ').strip()
    return ' '.join(text.split())

def fix_slangwords(text, slang_dict):
    words = text.split()
    return ' '.join([slang_dict.get(word, word) for word in words])

def stemming_text_func(text):
    return stemmer.stem(text) if stemmer else text

def remove_stopwords(text):
    return stopword_remover.remove(text) if stopword_remover else text

# --- Load assets on startup ---
load_ml_assets()
slangwords = load_slangwords(SLANGWORDS_PATH)
# Additional stopwords can be added here if needed
# all_stopwords = ...

try:
    df = pd.read_csv(CBR_DATA_PATH)
    texts_for_tfidf = df["combined_text"].fillna("").tolist()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts_for_tfidf)
    beach_data_for_search = df.to_dict(orient="records")
    logging.info("TF-IDF Vectorizer fitted and beach data loaded.")
except Exception as e:
    logging.critical(f"FATAL ERROR initializing TF-IDF and beach data: {e}", exc_info=True)
    tfidf_vectorizer = None
    tfidf_matrix = None
    beach_data_for_search = None


# =================================================================================
# --- REFACTORED AND NEW HELPER FUNCTIONS ---
# =================================================================================

def _preprocess_query(text):
    """Helper function to preprocess a text query."""
    text = cleaning_text(text)
    text = fix_slangwords(text, slangwords)
    text = remove_stopwords(text)
    # Stemming is optional and can be slow, enable if needed
    # text = stemming_text_func(text) 
    return text

def _get_tfidf_recommendations(query, limit=10, page=1):
    """
    Calculates recommendations based on TF-IDF and supports pagination.
    Returns a dictionary with 'recommendations' and 'totalCount'.
    """
    if not all([query, tfidf_vectorizer, tfidf_matrix is not None, beach_data_for_search]):
        return {"recommendations": [], "totalCount": 0}

    processed_query = _preprocess_query(query)
    if not processed_query.strip():
        logging.info("Query became empty after processing.")
        return {"recommendations": [], "totalCount": 0}

    user_tfidf = tfidf_vectorizer.transform([processed_query])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Get all relevant results (score > 0) and sort them
    relevant_indices = [i for i, score in enumerate(cosine_sim) if score > 0]
    sorted_relevant_indices = sorted(relevant_indices, key=lambda i: cosine_sim[i], reverse=True)
    
    total_count = len(sorted_relevant_indices)
    
    # Apply pagination
    start_index = (page - 1) * limit
    end_index = start_index + limit
    paginated_indices = sorted_relevant_indices[start_index:end_index]

    recommendations = []
    for i in paginated_indices:
        recommendations.append({
            "placeId": beach_data_for_search[i]['place_id'],
            "similarity_score": float(cosine_sim[i])
        })

    return {"recommendations": recommendations, "totalCount": total_count}

# =================================================================================
# --- API ENDPOINTS ---
# =================================================================================

@app.route('/search-point', methods=['GET'])
def search_point():
    """Handles search requests using TF-IDF with pagination."""
    if tfidf_vectorizer is None:
        return jsonify({"error": "Search model is not available."}), 503

    user_query = request.args.get('query')
    limit = int(request.args.get('limit', 10))
    page = int(request.args.get('page', 1))

    if not user_query:
        return jsonify({"error": "The 'query' parameter is required."}), 400

    try:
        logging.info(f"Searching for query: '{user_query}', page: {page}, limit: {limit}")
        result = _get_tfidf_recommendations(user_query, limit, page)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error during search-point: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during search."}), 500

@app.route('/recommend-beach', methods=['POST'])
def recommend_beach():
    """Handles personalized or general recommendations."""
    request_data = request.get_json()
    user_id = request_data.get('user_id')
    preference_text = request_data.get('preference_text')
    top_n = int(request_data.get('top_n', 10))

    # --- Skenario 1: Rekomendasi Terpersonalisasi ---
    if user_id and hybrid_recommender_interpreter and user_encoder and place_encoder:
        try:
            user_id_for_encoder = str(user_id).strip()
            if user_id_for_encoder not in user_encoder.classes_:
                logging.warning(f"User ID '{user_id_for_encoder}' not in encoder. Falling back.")
                return _handle_general_recommendation(preference_text, top_n)

            user_encoded_idx = user_encoder.transform([user_id_for_encoder])[0]
            
            all_place_ids = [item['place_id'] for item in beach_data_for_search]
            mappable_place_ids = [pid for pid in all_place_ids if pid in place_encoder.classes_]
            encoded_place_ids = place_encoder.transform(mappable_place_ids)

            input_details = hybrid_recommender_interpreter.get_input_details()
            output_details = hybrid_recommender_interpreter.get_output_details()
            
            user_input_array = np.full(len(encoded_place_ids), user_encoded_idx, dtype=np.int32)
            place_input_array = np.array(encoded_place_ids, dtype=np.int32)
            
            hybrid_recommender_interpreter.set_tensor(input_details[0]['index'], user_input_array)
            hybrid_recommender_interpreter.set_tensor(input_details[1]['index'], place_input_array)
            hybrid_recommender_interpreter.invoke()
            
            predictions = hybrid_recommender_interpreter.get_tensor(output_details[0]['index']).flatten()
            
            results = [{"placeId": pid, "score": float(score)} for pid, score in zip(mappable_place_ids, predictions)]
            results.sort(key=lambda x: x['score'], reverse=True)

            final_results = results[:top_n]
            return jsonify({"recommendations": final_results, "totalCount": len(final_results)})

        except Exception as e:
            logging.error(f"Personalized recommendation failed for user {user_id}: {e}. Falling back.", exc_info=True)
            # Fallback to general recommendation on any error
            return _handle_general_recommendation(preference_text, top_n)

    # --- Skenario 2: Rekomendasi Umum ---
    return _handle_general_recommendation(preference_text, top_n)

def _handle_general_recommendation(preference_text, top_n):
    """Handles general recommendations (content-based or popular)."""
    if preference_text:
        logging.info("Handling general recommendation with preference text.")
        result = _get_tfidf_recommendations(preference_text, limit=top_n, page=1)
        if not result["recommendations"]:
            logging.info("TF-IDF result empty, falling back to popular.")
            return _get_popular_beaches_from_data(top_n)
        return jsonify(result)
    else:
        logging.info("No preference text, returning popular beaches.")
        return _get_popular_beaches_from_data(top_n)

def _get_popular_beaches_from_data(top_n):
    """Returns top N popular beaches based on rating."""
    if not beach_data_for_search:
        return jsonify({"recommendations": [], "totalCount": 0})
    
    top_rated = sorted(beach_data_for_search, key=lambda x: x.get('rating', 0), reverse=True)
    recommendations = [{"placeId": b['place_id'], "score": float(b.get('rating', 0))} for b in top_rated[:top_n]]
    return jsonify({"recommendations": recommendations, "totalCount": len(recommendations)})


@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    """Handles sentiment analysis requests."""
    if not sentiment_model or not sentiment_tokenizer:
        return jsonify({"error": "Sentiment model is not available."}), 503

    data = request.get_json()
    review_text = data.get('review_text')
    if not review_text or not isinstance(review_text, str):
        return jsonify({"error": "No valid 'review_text' provided."}), 400

    try:
        processed_text = _preprocess_query(review_text) # Re-use preprocessing
        sequences = sentiment_tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

        if padded.size == 0 or np.all(padded == 0):
            return jsonify({"sentiment": "Neutral", "confidence": 0.5})

        prediction = sentiment_model.predict(padded)[0][0]
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        return jsonify({"sentiment": sentiment, "confidence": float(prediction)})
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during analysis."}), 500


if __name__ == '__main__':
    # Use environment variable for port, with a default, for deployment flexibility
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False for production