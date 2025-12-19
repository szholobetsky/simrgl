import sqlite3
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
import math

# Import configuration
try:
    import config
    DB_PATH = config.DB_FILE
    OUTPUT_FILE_TFIDF = config.WORD_GROUPS_WITH_TFIDF_OUTPUT
    INPUT_FILE_GROUPS = config.WORD_GROUPS_INPUT
except ImportError:
    # Fallback if config.py is not available
    DB_PATH = '../data/sonar.db'
    OUTPUT_FILE_TFIDF = "full_word_group_with_tfidf.csv"
    INPUT_FILE_GROUPS = "full_word_group.csv"

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

def preprocess_text(text, remove_stopwords=True):
    """Попередня обробка тексту [5-7]"""
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ')
    tokens = word_tokenize(text)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        return ' '.join(filtered_tokens)
    else:
        filtered_tokens = [word for word in tokens if len(word) > 1]
        return ' '.join(filtered_tokens)

def get_raw_tokens(text):
    """Отримання всіх токенів без фільтрації для TF/DF"""
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ')
    tokens = word_tokenize(text)
    return [word for word in tokens if len(word) > 0]

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

    df_tasks = get_task_data(conn)
    if df_tasks is None or df_tasks.empty:
        print("Не вдалося отримати дані з бази даних або дані відсутні")
        conn.close()
        return

    # 1. Підготовка документів для TF-IDF
    documents = []
    raw_documents_tokens = []
    for _, row in tqdm(df_tasks.iterrows(), total=len(df_tasks), desc="Підготовка документів для TF-IDF"):
        text = f"{row['TITLE']} {row['DESCRIPTION']} {row['COMMENTS']}"
        documents.append(preprocess_text(text))
        raw_text = f"{row['TITLE']} {row['DESCRIPTION']} {row['COMMENTS']}"
        raw_documents_tokens.append(get_raw_tokens(raw_text))

    # 2. Ініціалізація та навчання TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(documents)
    tfidf_matrix = tfidf_vectorizer.transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Створення словника TF-IDF для кожного слова
    tfidf_scores = {}
    for col in tfidf_matrix.T.tocsr():
        #print(col)
        #print(col.indices)
        #print(feature_names[col.indices])
        #tfidf_scores[feature_names[col.indices]] = np.mean(col.data) # Візьмемо середнє TF-IDF по всіх документах
        words = ' '.join(feature_names[col.indices])  # Перетворюємо масив слів на рядок
        tfidf_scores[words] = np.mean(col.data) # Візьмемо середнє TF-IDF по всіх документах

    # 3. Завантаження DataFrame з файлу "full_word_group.csv"
    try:
        df_grouped = pd.read_csv(INPUT_FILE_GROUPS, encoding='utf-8')
    except FileNotFoundError:
        print(f"Помилка: Файл '{INPUT_FILE_GROUPS}' не знайдено.")
        conn.close()
        return

    # 4. Обчислення TF та DF для кожного слова
    word_counts = Counter()
    document_frequency = Counter()
    for tokens in raw_documents_tokens:
        word_counts.update(tokens)
        unique_tokens_in_doc = set(tokens)
        document_frequency.update(unique_tokens_in_doc)

    total_documents = len(raw_documents_tokens)
    idf_scores = {}
    for word, df in document_frequency.items():
        idf_scores[word] = math.log(total_documents / (df + 1)) # Додаємо 1 для уникнення ділення на нуль

    # 5. Додавання TF, DF та TF-IDF до завантаженого DataFrame
    tf_list = []
    df_list = []
    tfidf_list = []
    idf_list = []

    for token in df_grouped['token']:
        tf_list.append(word_counts.get(token, 0))
        df_list.append(document_frequency.get(token, 0))
        idf_list.append(idf_scores.get(token, 0))
        tfidf_list.append(tfidf_scores.get(token, 0) if token in tfidf_scores else 0)

    df_grouped['TF'] = tf_list
    df_grouped['DF'] = df_list
    df_grouped['IDF'] = idf_list
    df_grouped['TFIDF'] = tfidf_list

    # 6. Збереження оновленого DataFrame у новий CSV файл
    try:
        df_grouped.to_csv(OUTPUT_FILE_TFIDF, index=False, encoding='utf-8')
        print(f"\nDataFrame з TF, DF, IDF та TF-IDF збережено у файл: {OUTPUT_FILE_TFIDF}")
    except Exception as e:
        print(f"Помилка при збереженні DataFrame у файл: {e}")

    conn.close()

if __name__ == "__main__":
    main()
