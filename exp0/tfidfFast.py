import sqlite3
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Import configuration
try:
    import config
    DB_PATH = config.DB_FILE
except ImportError:
    # Fallback if config.py is not available
    DB_PATH = '../data/sonar.db'

def connect_to_db(db_path):
    """Підключення до SQLite бази даних [1]"""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Помилка при підключенні до бази даних: {e}")
        return None

def get_task_data(conn):
    """Отримання даних з таблиці TASK [2-4]"""
    query = """
    SELECT TITLE, DESCRIPTION, COMMENTS
    FROM TASK
    WHERE TITLE IS NOT NULL OR DESCRIPTION IS NOT NULL OR COMMENTS IS NOT NULL
    """
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except pd.io.sql.DatabaseError as e:
        print(f"Помилка при запиті даних: {e}")
        return None

def preprocess_text(text):
    """Попередня обробка тексту [5-7]"""
    if pd.isna(text):
        return '' # Повертаємо пустий рядок, щоб не було помилок при об'єднанні
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ')
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(filtered_tokens) # Повертаємо об'єднаний рядок для TfidfVectorizer

def main():
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        word_tokenize("example")
    except LookupError:
        nltk.download('punkt')

    conn = connect_to_db(DB_PATH)
    if not conn:
        return

    df = get_task_data(conn)
    if df is None or df.empty:
        print("Не вдалося отримати дані з бази даних або дані відсутні")
        conn.close()
        return

    # Створення документів для TF-IDF
    documents = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Підготовка документів"):
        text = f"{row['TITLE']} {row['DESCRIPTION']} {row['COMMENTS']}"
        documents.append(preprocess_text(text))

    # Ініціалізація TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Обчислення TF-IDF
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Отримання словника термінів
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Створення DataFrame з результатами TF-IDF
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    print("\nМатриця TF-IDF (перші 5 рядків):")
    print(tfidf_df.head())

    # Ви також можете зберегти цю матрицю у CSV файл
    tfidf_df.to_csv("tfidf_matrix.csv", index=False, encoding='utf-8')
    print("\nМатрицю TF-IDF збережено у файл: tfidf_matrix.csv")

    conn.close()

if __name__ == "__main__":
    main()
