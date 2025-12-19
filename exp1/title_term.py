import sqlite3
import re
from collections import Counter

# --- Налаштування ---
DB_FILE = "flink_mobile.db"  # Назва вашого файлу бази даних
TASKS_TABLE = "TASK"
TASK_NAME_COLUMN = "NAME"
TASK_TITLE_COLUMN = "TITLE"

# Назви таблиць, як у першому запиті
TITLE_TERM_TABLE = "TITLE_TERM"
TITLE_TASK_TERM_TABLE = "TITLE_TASK_TERM"
TITLE_TASK_TERM_AGG_TABLE = "TITLE_TASK_TERM_AGG"

def create_and_populate_tables(db_file):
    """
    Створює та заповнює таблиці TITLE_TERM, TITLE_TASK_TERM, TITLE_TASK_TERM_AGG.
    Використовуючи поля NAME та TITLE з таблиці TASK.
    """
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        print("Підключено до бази даних.")

        # --- Створення таблиць з правильними назвами та полями ---
        print("Створення таблиць...")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TITLE_TERM_TABLE} (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                TERM TEXT NOT NULL UNIQUE
            );
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TITLE_TASK_TERM_TABLE} (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                TASK_NAME TEXT NOT NULL,
                TERM_ID INTEGER NOT NULL,
                FOREIGN KEY (TERM_ID) REFERENCES {TITLE_TERM_TABLE}(ID)
            );
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TITLE_TASK_TERM_AGG_TABLE} (
                TASK_NAME TEXT NOT NULL,
                TERM_ID INTEGER NOT NULL,
                CNT INTEGER NOT NULL,
                PRIMARY KEY (TASK_NAME, TERM_ID),
                FOREIGN KEY (TERM_ID) REFERENCES {TITLE_TERM_TABLE}(ID)
            );
        """)
        conn.commit()
        print("Таблиці створено (якщо не існували).")

        # --- Отримання даних з таблиці TASK з правильними іменами полів ---
        print("Зчитування даних з таблиці TASK...")
        cursor.execute(f"SELECT {TASK_NAME_COLUMN}, {TASK_TITLE_COLUMN} FROM {TASKS_TABLE}")
        tasks = cursor.fetchall()

        if not tasks:
            print(f"Таблиця {TASKS_TABLE} порожня. Завершення.")
            return

        # --- Обробка та заповнення таблиць ---
        all_terms_map = {}  # Словник для кешування термінів: {термін: ID}
        
        # Обробка кожного завдання
        for task_name, title in tasks:
            if not title:
                continue

            # Нормалізація тексту: переведення в нижній регістр та видалення розділових знаків
            normalized_title = re.sub(r'[^a-zA-Z0-9\s-]', '', title).lower()
            terms = normalized_title.split()
            
            if not terms:
                continue

            # Заповнення TITLE_TERM та TITLE_TASK_TERM
            for term in terms:
                if term not in all_terms_map:
                    cursor.execute(f"SELECT ID FROM {TITLE_TERM_TABLE} WHERE TERM = ?", (term,))
                    term_id = cursor.fetchone()
                    if not term_id:
                        cursor.execute(f"INSERT INTO {TITLE_TERM_TABLE} (TERM) VALUES (?)", (term,))
                        term_id = cursor.lastrowid
                    else:
                        term_id = term_id[0]
                    all_terms_map[term] = term_id
                else:
                    term_id = all_terms_map[term]
                
                cursor.execute(f"INSERT INTO {TITLE_TASK_TERM_TABLE} (TASK_NAME, TERM_ID) VALUES (?, ?)", (task_name, term_id))
            
            # Заповнення TITLE_TASK_TERM_AGG
            term_counts = Counter(terms)
            for term, count in term_counts.items():
                term_id = all_terms_map[term]
                cursor.execute(f"INSERT OR REPLACE INTO {TITLE_TASK_TERM_AGG_TABLE} (TASK_NAME, TERM_ID, CNT) VALUES (?, ?, ?)", (task_name, term_id, count))
        
        conn.commit()
        print("Дані успішно заповнено.")

    except sqlite3.Error as e:
        print(f"Помилка SQLite: {e}")
    finally:
        if conn:
            conn.close()
            print("З'єднання з базою даних закрито.")

if __name__ == "__main__":
    create_and_populate_tables(DB_FILE)
