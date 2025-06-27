import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text, remove_stopwords=True):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = text.split()

    if remove_stopwords:
        words = [w for w in words if w not in stop_words]

    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)
