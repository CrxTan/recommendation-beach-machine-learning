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
CORS(app) # Mengaktifkan CORS untuk semua endpoint

# --- Konfigurasi Path Model ---
BASE_DIR = os.path.dirname(__file__)

# Sentiment Models Paths (Tidak berubah, masih menggunakan .h5)
SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'sentiment_model_lstm.h5')
SENTIMENT_TOKENIZER_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'tokenizer.pkl')
LEXICON_POSITIVE_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'lexicon_positive.json')
LEXICON_NEGATIVE_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'lexicon_negative.json')
SLANGWORDS_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'combined_slang_words.txt')
STOPWORDS_PATH = os.path.join(BASE_DIR, 'Models', 'sentimen', 'combined_stop_words.txt')

# Search/Content-Based Filtering Models Paths
CBR_DATA_PATH = os.path.join(BASE_DIR, 'Models', 'search', 'cbr_clean.csv')

# --- PERUBAHAN DI SINI: Menggunakan model .tflite ---
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'recommend', 'model.tflite')
PLACE_ENCODER_PATH = os.path.join(BASE_DIR, 'Models', 'recommend', 'place_encoder.pkl')
USER_ENCODER_PATH = os.path.join(BASE_DIR, 'Models', 'recommend', 'user_encoder.pkl')

# --- Global Variables for ML Assets ---
sentiment_model = None
sentiment_tokenizer = None
lexicon_positive = {}
lexicon_negative = {}
stemmer = None

# For Search/Content-Based Filtering
tfidf_vectorizer = None
tfidf_matrix = None
beach_data_for_search = None

# For Hybrid Recommender
# PERUBAHAN: Gunakan interpreter, bukan model langsung
hybrid_recommender_interpreter = None
place_encoder = None
user_encoder = None
encoded_place_to_original_id_map = {}

# Preprocessing assets that are loaded/initialized globally
stopword_remover = None
slangwords = {}
all_stopwords = set()

MAX_SEQUENCE_LENGTH = 10

# --- Fungsi untuk Memuat Aset ML ---
def load_ml_assets():
    """Memuat semua model ML dan aset lainnya saat aplikasi dimulai."""
    global sentiment_model, sentiment_tokenizer, lexicon_positive, lexicon_negative, stemmer, \
           hybrid_recommender_interpreter, place_encoder, user_encoder, \
           slangwords, all_stopwords, encoded_place_to_original_id_map, stopword_remover

    logging.info("Starting to load ML assets...")
    try:
        # 1. Load Sentiment Models (Tidak berubah)
        logging.info(f"Loading sentiment model from: {SENTIMENT_MODEL_PATH}")
        sentiment_model = tf.keras.models.load_model(SENTIMENT_MODEL_PATH)
        logging.info("Sentiment model loaded successfully.")

        logging.info(f"Loading sentiment tokenizer from: {SENTIMENT_TOKENIZER_PATH}")
        with open(SENTIMENT_TOKENIZER_PATH, 'rb') as f:
            sentiment_tokenizer = pickle.load(f)
        logging.info("Sentiment Tokenizer loaded successfully.")

        logging.info(f"Loading positive lexicon from: {LEXICON_POSITIVE_PATH}")
        with open(LEXICON_POSITIVE_PATH, 'r', encoding='utf-8') as f:
            lexicon_positive = json.load(f)
        logging.info("Positive lexicon loaded.")

        logging.info(f"Loading negative lexicon from: {LEXICON_NEGATIVE_PATH}")
        with open(LEXICON_NEGATIVE_PATH, 'r', encoding='utf-8') as f:
            lexicon_negative = json.load(f)
        logging.info("Negative lexicon loaded.")

        # 2. Load Sastrawi Stemmer (Tidak berubah)
        logging.info("Loading Sastrawi Stemmer...")
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        logging.info("Sastrawi Stemmer loaded.")
        
        stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        logging.info("Sastrawi StopWordRemover initialized.")

        # --- PERUBAHAN DI SINI: Menggunakan TFLite Interpreter ---
        logging.info(f"Loading Hybrid Recommender Model from: {TFLITE_MODEL_PATH}")
        hybrid_recommender_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        hybrid_recommender_interpreter.allocate_tensors()
        logging.info("Hybrid Recommender Model (TFLite) loaded successfully.")

        # Load Encoders (Tidak berubah)
        logging.info(f"Loading Place Encoder from: {PLACE_ENCODER_PATH}")
        with open(PLACE_ENCODER_PATH, 'rb') as f:
            place_encoder = pickle.load(f)
        logging.info("Place Encoder loaded successfully.")

        logging.info(f"Loading User Encoder from: {USER_ENCODER_PATH}")
        with open(USER_ENCODER_PATH, 'rb') as f:
            user_encoder = pickle.load(f)
        logging.info("User Encoder loaded successfully.")

        if place_encoder and hasattr(place_encoder, 'classes_'):
            encoded_place_to_original_id_map = {
                i: place_id for i, place_id in enumerate(place_encoder.classes_)
            }
            logging.info("Place Encoder map created.")
        else:
            logging.warning("Warning: place_encoder does not have 'classes_' attribute. Cannot create mapping.")

    except FileNotFoundError as e:
        logging.error(f"FATAL ERROR: One or more ML asset files not found. Please ensure all files are in the correct directories.")
        logging.error(f"Missing file: {e.filename}")
        sentiment_model = None
        sentiment_tokenizer = None
        hybrid_recommender_interpreter = None # PERUBAHAN
        place_encoder = None
        user_encoder = None
    except Exception as e:
        logging.critical(f"FATAL ERROR: General error loading ML assets: {e}", exc_info=True)
        sentiment_model = None
        sentiment_tokenizer = None
        hybrid_recommender_interpreter = None # PERUBAHAN
        place_encoder = None
        user_encoder = None
    logging.info("Finished loading ML assets.")


# --- Fungsi Preprocessing --- (Tidak berubah)
def load_slangwords(file_path):
    slangwords_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.endswith('.json'):
                slangwords_dict = json.load(file)
            else:
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

# Panggil load_ml_assets() saat startup
load_ml_assets()

try:
    slangwords = load_slangwords(SLANGWORDS_PATH)
    indonesia_stopwords_set = load_stopwords(STOPWORDS_PATH)
    english_stopwords_list = ["the", "a", "an", "is", "of", "and", "to", "in", "for", "with", "on"]
    custom_stopwords_list = ["iya", "yaa", "gak", "nya", "na", "sih", "ku", "di", "ga", "ya", "gaa", "loh", "kah", "woi", "woii", "woy", "banget", "oke"]
    all_stopwords = set(list(indonesia_stopwords_set) + english_stopwords_list + custom_stopwords_list)
    logging.info("All stopwords combined.")

    df = pd.read_csv(CBR_DATA_PATH)
    texts_for_tfidf = df["combined_text"].fillna("").tolist()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts_for_tfidf)
    logging.info("TF-IDF Vectorizer fitted and TF-IDF Matrix created.")

    beach_data_for_search = df.to_dict(orient="records")
    logging.info("Beach data for search loaded and converted to list of dicts.")

except Exception as e:
    logging.critical(f"FATAL ERROR: Error initializing TF-IDF and beach data: {e}", exc_info=True)
    tfidf_vectorizer = None
    tfidf_matrix = None
    beach_data_for_search = None


def cleaning_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace(r'@[\w\d]+', ' ')
    text = text.replace(r'#[\w\d]+', ' ')
    text = text.replace(r'RT[\s]', ' ')
    text = text.replace(r'http\S+|www\S+', ' ')
    text = text.replace(r'\d+', ' ')
    text = text.replace(r'[^a-zA-Z\s]', ' ')
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    text = text.strip()
    return text

def casefolding_text(text):
    return text.lower()

def fix_slangwords(text):
    if not slangwords:
        logging.warning("Slangwords dictionary is empty. Skipping slang word fixing.")
        return text
    words = text.split()
    fixed_words = [slangwords.get(word, word) for word in words]
    return ' '.join(fixed_words)

def tokenizing_text(text, current_tokenizer):
    if current_tokenizer is None:
        logging.error("Tokenizer is None. Cannot tokenize text.")
        return []
    sequences = current_tokenizer.texts_to_sequences([text])
    if sequences and len(sequences[0]) > 0:
        return sequences[0]
    return []

def filtering_text(tokens):
    if not all_stopwords:
        logging.warning("Stopwords set is empty. Skipping stopword filtering.")
        return tokens
    return [word for word in tokens if word not in all_stopwords]

def stemming_text_func(text):
    if stemmer:
        return stemmer.stem(text)
    logging.warning("Stemmer is None. Skipping stemming.")
    return text

# --- Endpoint for Sentiment Analysis --- (Tidak berubah)
@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    if sentiment_model is None or sentiment_tokenizer is None:
        logging.error("Sentiment ML models not loaded. Returning 500.")
        return jsonify({"error": "Sentiment ML models not loaded or still initializing."}), 500

    data = request.get_json()
    review_text = data.get('review_text')

    if not review_text or not isinstance(review_text, str):
        logging.warning(f"Invalid or no review_text provided: {review_text}. Returning 400.")
        return jsonify({"error": "No valid review_text provided"}), 400

    try:
        logging.info(f"Analyzing sentiment for: '{review_text[:50]}...'")
        processed_text = cleaning_text(review_text)
        logging.debug(f"Cleaned: '{processed_text}'")
        processed_text = casefolding_text(processed_text)
        logging.debug(f"Casefolded: '{processed_text}'")
        processed_text = fix_slangwords(processed_text)
        logging.debug(f"Slang fixed: '{processed_text}'")
        filtered_words = filtering_text(processed_text.split())
        logging.debug(f"Filtered words: {filtered_words}")
        stemmed_text = stemming_text_func(' '.join(filtered_words))
        logging.debug(f"Stemmed text: '{stemmed_text}'")
        sequences = sentiment_tokenizer.texts_to_sequences([stemmed_text])
        logging.debug(f"Tokenized sequences: {sequences}")
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        logging.debug(f"Padded sequences shape: {padded_sequences.shape}")

        if padded_sequences.size == 0 or np.all(padded_sequences == 0):
            logging.info("Processed text resulted in empty or all-zero padded sequence. Returning Neutral.")
            return jsonify({"sentiment": "Neutral", "confidence": 0.5})

        prediction = sentiment_model.predict(padded_sequences)[0][0]
        sentiment_label = "Positive" if prediction >= 0.5 else "Negative"
        logging.info(f"Prediction: {prediction}, Sentiment: {sentiment_label}")

        return jsonify({
            "sentiment": sentiment_label,
            "confidence": float(prediction)
        })

    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}", exc_info=True)
        return jsonify({"error": f"Failed to analyze sentiment due to internal error: {str(e)}"}), 500

# --- Endpoint for Content-Based Search (using TF-IDF) --- (Tidak berubah)
@app.route('/search-point', methods=['POST'])
def search_point():
    if tfidf_vectorizer is None or tfidf_matrix is None or beach_data_for_search is None:
        logging.error("TF-IDF model or beach data not loaded for search-point. Returning 500.")
        return jsonify({"error": "Search model or data not loaded on server."}), 500

    req_data = request.get_json()
    user_input = req_data.get('query')
    top_n = req_data.get('top_n', 10)

    if not user_input or not isinstance(user_input, str):
        logging.warning(f"Invalid search input provided: {user_input}. Returning 400.")
        return jsonify({"error": "Invalid search input. 'query' must be a non-empty string."}), 400

    try:
        logging.info(f"Searching for query: '{user_input[:50]}...'")
        user_input_processed = cleaning_text(user_input)
        user_input_processed = casefolding_text(user_input_processed)
        user_input_processed = fix_slangwords(user_input_processed)
        
        if stopword_remover:
            user_input_processed = stopword_remover.remove(user_input_processed)
        else:
            logging.warning("Stopword remover is not initialized. Skipping stopword removal for search query.")
            words_after_slang = user_input_processed.split()
            filtered_words = filtering_text(words_after_slang)
            user_input_processed = ' '.join(filtered_words)

        if not user_input_processed.strip():
            logging.info("Search query became empty after processing. Returning no recommendations.")
            return jsonify({"recommendations": []})

        user_tfidf = tfidf_vectorizer.transform([user_input_processed])
        cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

        if np.max(cosine_sim) == 0:
            logging.info("No similar items found for the search query.")
            return jsonify({"recommendations": []})

        top_indices = cosine_sim.argsort()[-top_n:][::-1]

        recommendations = []
        for i in top_indices:
            if 0 <= i < len(beach_data_for_search):
                beach_info = beach_data_for_search[i]
                recommendations.append({
                    "placeId": beach_info['place_id'],
                    "place_name": beach_info.get('place_name', 'N/A'),
                    "description": beach_info.get('description', 'No description available'),
                    "rating": beach_info.get('rating', 0),
                    "featured_image": beach_info.get('featured_image', 'N/A'),
                    "similarity_score": float(cosine_sim[i])
                })
        logging.info(f"Found {len(recommendations)} recommendations for query '{user_input[:50]}...'")
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        logging.error(f"Error during search-point recommendation: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing the search request: {str(e)}"}), 500

# --- Endpoint for Hybrid Recommendation (Personalized vs. General/Popular) ---
@app.route('/recommend-beach', methods=['POST'])
def recommend_beach():
    request_data = request.get_json()
    user_id_from_node = request_data.get('user_id')
    preference_text = request_data.get('preference_text')
    top_n = request_data.get('top_n', 10)

    # --- Skenario 1: Rekomendasi Terpersonalisasi ---
    # PERUBAHAN: Cek interpreter, bukan model
    if user_id_from_node and hybrid_recommender_interpreter and user_encoder and place_encoder and beach_data_for_search:
        try:
            user_id_for_encoder = str(user_id_from_node).strip()
            
            if user_id_for_encoder not in user_encoder.classes_:
                logging.warning(f"User ID '{user_id_for_encoder}' not found in user_encoder. Falling back to general recommendation.")
                return _handle_general_recommendation(preference_text, top_n)

            user_encoded_idx = user_encoder.transform([user_id_for_encoder])[0]
            logging.info(f"Personalized recommendation for user_id: {user_id_from_node}, encoded_idx: {user_encoded_idx}")

            all_original_place_ids = [item['place_id'] for item in beach_data_for_search]
            mappable_place_ids_original = [
                pid for pid in all_original_place_ids if pid in place_encoder.classes_
            ]
            
            if not mappable_place_ids_original:
                logging.warning("No mappable place IDs found for personalized prediction. Falling back to general.")
                return _handle_general_recommendation(preference_text, top_n)

            encoded_place_ids_for_prediction = place_encoder.transform(mappable_place_ids_original)
            
            # --- PERUBAHAN DI SINI: Prediksi menggunakan TFLite Interpreter ---
            
            # Dapatkan input dan output tensor dari interpreter
            input_details = hybrid_recommender_interpreter.get_input_details()
            output_details = hybrid_recommender_interpreter.get_output_details()
            
            # Buat array input yang sesuai dengan format model tflite
            user_input_array = np.full(len(encoded_place_ids_for_prediction), user_encoded_idx, dtype=np.int32)
            place_input_array = np.array(encoded_place_ids_for_prediction, dtype=np.int32)
            
            # Set nilai input pada tensor interpreter
            hybrid_recommender_interpreter.set_tensor(input_details[0]['index'], user_input_array)
            hybrid_recommender_interpreter.set_tensor(input_details[1]['index'], place_input_array)
            
            # Jalankan inferensi
            hybrid_recommender_interpreter.invoke()
            
            # Ambil hasil prediksi dari tensor output
            predictions = hybrid_recommender_interpreter.get_tensor(output_details[0]['index']).flatten()
            
            logging.debug(f"Predictions shape: {predictions.shape}")

            personalized_results = []
            for i, score in enumerate(predictions):
                original_place_id = mappable_place_ids_original[i]
                personalized_results.append({
                    "placeId": original_place_id,
                    "score": float(score)
                })

            personalized_results.sort(key=lambda x: x['score'], reverse=True)
            logging.info(f"Generated {len(personalized_results[:top_n])} personalized recommendations for user {user_id_from_node}.")
            return jsonify({"recommendations": personalized_results[:top_n]})

        except ValueError as ve:
            logging.error(f"ValueError in personalized recommendation for user {user_id_from_node}: {ve}. Falling back to general.", exc_info=True)
            return _handle_general_recommendation(preference_text, top_n)
        except Exception as e:
            logging.error(f"Error in personalized recommendation for user {user_id_from_node}: {e}. Falling back to general.", exc_info=True)
            return _handle_general_recommendation(preference_text, top_n)

    # --- Skenario 2: Rekomendasi Umum (Tidak berubah) ---
    else:
        logging.info(f"User is not logged in or personalization model not available/failed. Performing general recommendation.")
        return _handle_general_recommendation(preference_text, top_n)

def _handle_general_recommendation(preference_text, top_n):
    if preference_text and tfidf_vectorizer and tfidf_matrix is not None and beach_data_for_search is not None:
        logging.info("Generating content-based recommendation using preference_text.")
        try:
            user_input_processed = cleaning_text(preference_text)
            user_input_processed = casefolding_text(user_input_processed)
            user_input_processed = fix_slangwords(user_input_processed)
            
            if stopword_remover:
                user_input_processed = stopword_remover.remove(user_input_processed)
            else:
                words_after_slang = user_input_processed.split()
                filtered_words = filtering_text(words_after_slang)
                user_input_processed = ' '.join(filtered_words)

            if not user_input_processed.strip():
                logging.info("Preference text became empty after processing. Falling back to popular.")
                return _get_popular_beaches_from_data(top_n)

            user_tfidf = tfidf_vectorizer.transform([user_input_processed])
            cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

            if np.max(cosine_sim) == 0:
                logging.info("No similarity found for preference text. Falling back to popular.")
                return _get_popular_beaches_from_data(top_n)

            top_indices = cosine_sim.argsort()[-top_n:][::-1]

            recommendations = []
            for i in top_indices:
                if 0 <= i < len(beach_data_for_search):
                    recommendations.append({
                        "placeId": beach_data_for_search[i]['place_id'],
                        "score": float(cosine_sim[i])
                    })
            logging.info(f"Generated {len(recommendations)} content-based recommendations.")
            return jsonify({"recommendations": recommendations})
        except Exception as e:
            logging.error(f"Error during content-based recommendation: {e}. Falling back to popular.", exc_info=True)
            return _get_popular_beaches_from_data(top_n)
    else:
        logging.info("Generating general recommendations (top-rated/popular).")
        return _get_popular_beaches_from_data(top_n)

def _get_popular_beaches_from_data(top_n):
    if beach_data_for_search:
        top_rated_beaches = sorted(beach_data_for_search, key=lambda x: x.get('rating', 0), reverse=True)
        general_recommendations = []
        for i in range(min(top_n, len(top_rated_beaches))):
            general_recommendations.append({
                "placeId": top_rated_beaches[i]['place_id'],
                "score": float(top_rated_beaches[i].get('rating', 0) / 5.0)
            })
        logging.info(f"Generated {len(general_recommendations)} popular recommendations.")
        return jsonify({"recommendations": general_recommendations})
    else:
        logging.error("No beach data available for popular recommendations. Returning empty list.")
        return jsonify({"recommendations": []})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)