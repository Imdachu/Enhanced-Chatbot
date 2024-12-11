import json
import random
import os
import traceback
import nltk
import streamlit as st

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Explicitly download all required NLTK resources
def download_nltk_resources():
    """
    Download all required NLTK resources with error handling
    """
    nltk_resources = [
        'punkt',      # Tokenization
        'wordnet',    # Lemmatization
        'stopwords',  # Stopwords list
        'averaged_perceptron_tagger'  # Part-of-speech tagging (optional but recommended)
    ]
    
    for resource in nltk_resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            st.warning(f"Could not download {resource} resource: {e}")

# Call the download function early
download_nltk_resources()

class EnhancedChatbot:
    def __init__(self, intents_file: str):
        """
        Initialize the chatbot with intents and preprocessing setup
        
        Args:
            intents_file (str): Path to the JSON file containing intents
        """
        # Initialize core components
        self.lemmatizer = WordNetLemmatizer()
        
        # Safely get stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            st.warning("Stopwords resource not available. Using minimal stopwords.")
            self.stopwords = {'the', 'a', 'an', 'in', 'to', 'for'}
        
        # Load intents with error handling
        try:
            with open(intents_file, 'r', encoding='utf-8') as file:
                self.intents = json.load(file)
        except FileNotFoundError:
            st.error(f"Intents file not found: {intents_file}")
            self.intents = {"intents": [
                {
                    "tag": "fallback",
                    "patterns": ["unknown"],
                    "responses": ["I'm not sure how to respond to that."]
                }
            ]}
        except json.JSONDecodeError:
            st.error(f"Invalid JSON in intents file: {intents_file}")
            self.intents = {"intents": [
                {
                    "tag": "fallback",
                    "patterns": ["unknown"],
                    "responses": ["I'm experiencing an issue understanding the intents."]
                }
            ]}
        
        # Prepare training data
        self.patterns = []
        self.tags = []
        
        # Robust pattern processing
        for intent in self.intents.get('intents', []):
            for pattern in intent.get('patterns', []):
                # Fallback tokenization if NLTK fails
                try:
                    tokens = nltk.word_tokenize(pattern.lower())
                except Exception:
                    tokens = pattern.lower().split()
                
                # Lemmatize and filter tokens
                processed_tokens = [
                    self.lemmatizer.lemmatize(token) 
                    for token in tokens 
                    if token not in self.stopwords and token.isalnum()
                ]
                
                self.patterns.append(' '.join(processed_tokens))
                self.tags.append(intent.get('tag', 'unknown'))
        
        # Handle empty patterns
        if not self.patterns:
            st.warning("No valid patterns found in intents file.")
            self.patterns = ['unknown']
            self.tags = ['fallback']
        
        # Create TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.patterns)

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text: tokenize, lemmatize, remove stopwords
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            str: Preprocessed text
        """
        # Fallback tokenization if NLTK fails
        try:
            tokens = nltk.word_tokenize(text.lower())
        except Exception:
            tokens = text.lower().split()
        
        # Lemmatize and filter stopwords
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stopwords and token.isalnum()
        ]
        
        return ' '.join(processed_tokens)

    def predict_intent(self, user_input: str) -> str:
        """
        Predict the intent of user input using cosine similarity
        
        Args:
            user_input (str): User's input text
        
        Returns:
            str: Predicted intent tag
        """
        try:
            preprocessed_input = self.preprocess_text(user_input)
            input_vector = self.vectorizer.transform([preprocessed_input])
            
            similarities = cosine_similarity(input_vector, self.tfidf_matrix)[0]
            best_match_index = similarities.argmax()
            
            # Adjust confidence threshold
            if similarities[best_match_index] > 0.2:  
                return self.tags[best_match_index]
            else:
                return "fallback"
        except Exception as e:
            st.error(f"Error in intent prediction: {e}")
            return "fallback"

    def get_response(self, intent: str) -> str:
        """
        Get a response based on the predicted intent
        
        Args:
            intent (str): Predicted intent tag
        
        Returns:
            str: Response message
        """
        for intent_data in self.intents.get('intents', []):
            if intent_data.get('tag') == intent:
                return random.choice(intent_data.get('responses', ["I'm not sure how to respond."]))
        return "I'm sorry, I didn't understand that."

    def chat(self, user_input: str) -> str:
        """
        Main chat method to process user input and generate response
        
        Args:
            user_input (str): User's input text
        
        Returns:
            str: Chatbot's response
        """
        intent = self.predict_intent(user_input)
        return self.get_response(intent)

def load_chat_history(file_path: str) -> List[Dict[str, str]]:
    """
    Load chat history from a JSON file
    
    Args:
        file_path (str): Path to the chat history file
    
    Returns:
        List[Dict[str, str]]: Chat history
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Error loading chat history: {e}")
    return []

def save_chat_history(file_path: str, chat_history: List[Dict[str, str]]) -> None:
    """
    Save chat history to a JSON file
    
    Args:
        file_path (str): Path to the chat history file
        chat_history (List[Dict[str, str]]): Chat history
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(chat_history, file, indent=4)
    except IOError as e:
        st.error(f"Error saving chat history: {e}")

def main():
    # Configure Streamlit page
    st.set_page_config(page_title="Enhanced Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 Enhanced Chatbot")
    st.markdown("An intelligent chatbot powered by NLP techniques")

    # Compute absolute paths
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        intents_path = os.path.join(script_dir, "intents.json")
        chat_history_path = os.path.join(script_dir, "chat_history.json")
    except Exception as e:
        st.error(f"Error determining file paths: {e}")
        return

    # Initialize chatbot with error handling
    try:
        chatbot = EnhancedChatbot(intents_path)
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return

    # Load previous chat history
    chat_history = load_chat_history(chat_history_path)

    # Sidebar for menu
    menu = ["Chat", "Conversation History"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Chat":
        st.subheader("Live Chat")
        user_input = st.chat_input("Type your message...")

        if user_input:
            try:
                # Add user message to chat history
                user_message = {"role": "user", "message": user_input}
                chat_history.append(user_message)
                st.chat_message("user").write(user_input)

                # Get chatbot response
                bot_response = chatbot.chat(user_input)
                bot_message = {"role": "assistant", "message": bot_response}
                chat_history.append(bot_message)
                st.chat_message("assistant").write(bot_response)

                # Save updated chat history
                save_chat_history(chat_history_path, chat_history)
            except Exception as e:
                st.error(f"Error processing chat: {e}")
                traceback.print_exc()

    elif choice == "Conversation History":
        st.subheader("Conversation History")

        if not chat_history:
            st.info("No conversation history available.")
        else:
            for idx, message in enumerate(chat_history):
                if message['role'] == 'user':
                    st.write(f"User: {message['message']}")
                else:
                    st.write(f"Assistant: {message['message']}")

if __name__ == "__main__":
    main()