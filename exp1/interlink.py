import sqlite3
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict

# --- Налаштування ---
DB_FILE = "data/sonar.db"
TASK_TABLE = "TASK"
TASK_NAME_COLUMN = "NAME"
TITLE_TASK_TERM_TABLE = "TITLE_TASK_TERM"
MODULE_TASK_TABLE = "MODULE_TASK"
MODULE_TABLE = "MODULE"

TERM_LINKS_TABLE = "TERM_LINKS"
FILE_LINKS_TABLE = "FILE_LINKS"

# Новий параметр для контролю пам'яті
BATCH_SIZE = 500  # Кількість завдань для обробки за один раз

def calculate_co_occurrence(db_file):
    """
    Розраховує та заповнює таблиці сумісності термінів та файлів.
    Використовує пакетну обробку для стабільності.
    """
    try:
        conn = sqlite3.connect(db_file)
        # Використовуємо буферизований курсор для великих запитів
        cursor = conn.cursor()
        
        print("Підключено до бази даних.")

        # --- Створення таблиць ---
        print("Створення таблиць для зв'язків...")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TERM_LINKS_TABLE} (
                L_TERM INTEGER NOT NULL,
                R_TERM INTEGER NOT NULL,
                CNT INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (L_TERM, R_TERM)
            );
        """)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {FILE_LINKS_TABLE} (
                L_FILE INTEGER NOT NULL,
                R_FILE INTEGER NOT NULL,
                CNT INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (L_FILE, R_FILE)
            );
        """)
        conn.commit()
        print("Таблиці створено (якщо не існували).")

        # --- Підготовка даних ---
        print("Підготовка даних для обробки...")
        cursor.execute(f"SELECT COUNT(DISTINCT {TASK_NAME_COLUMN}) FROM {TASK_TABLE}")
        total_tasks = cursor.fetchone()[0]

        cursor.execute(f"SELECT ID FROM {MODULE_TABLE} WHERE FILE = 1")
        file_module_ids = set(row[0] for row in cursor.fetchall())

        if total_tasks == 0:
            print("Таблиця TASK порожня. Завершення.")
            return

        print(f"Починаю аналіз {total_tasks} завдань...")
        
        # Обробка завдань пакетами
        offset = 0
        with tqdm(total=total_tasks, unit="завдань") as pbar:
            while offset < total_tasks:
                # Отримуємо наступний пакет завдань
                cursor.execute(f"SELECT DISTINCT {TASK_NAME_COLUMN} FROM {TASK_TABLE} LIMIT ? OFFSET ?", (BATCH_SIZE, offset))
                task_names_batch = [row[0] for row in cursor.fetchall()]

                term_links_counter = defaultdict(int)
                file_links_counter = defaultdict(int)

                for task_name in task_names_batch:
                    # Обробка термінів
                    cursor.execute(f"SELECT TERM_ID FROM {TITLE_TASK_TERM_TABLE} WHERE TASK_NAME = ? GROUP BY TERM_ID", (task_name,))
                    term_ids = sorted([row[0] for row in cursor.fetchall()])
                    
                    for term_pair in combinations(term_ids, 2):
                        term_links_counter[term_pair] += 1
                    
                    # Обробка файлів
                    cursor.execute(f"SELECT MODULE_ID FROM {MODULE_TASK_TABLE} WHERE TASK_NAME = ?", (task_name,))
                    module_ids_in_task = sorted([row[0] for row in cursor.fetchall() if row[0] in file_module_ids])

                    for file_pair in combinations(module_ids_in_task, 2):
                        file_links_counter[file_pair] += 1
                
                # --- Збереження результатів пакета ---
                term_links_to_insert = [(l_term, r_term, cnt) for (l_term, r_term), cnt in term_links_counter.items()]
                cursor.executemany(f"INSERT INTO {TERM_LINKS_TABLE} (L_TERM, R_TERM, CNT) VALUES (?, ?, ?) ON CONFLICT(L_TERM, R_TERM) DO UPDATE SET CNT = CNT + excluded.CNT", term_links_to_insert)

                file_links_to_insert = [(l_file, r_file, cnt) for (l_file, r_file), cnt in file_links_counter.items()]
                cursor.executemany(f"INSERT INTO {FILE_LINKS_TABLE} (L_FILE, R_FILE, CNT) VALUES (?, ?, ?) ON CONFLICT(L_FILE, R_FILE) DO UPDATE SET CNT = CNT + excluded.CNT", file_links_to_insert)
                
                conn.commit()
                
                offset += BATCH_SIZE
                pbar.update(len(task_names_batch))

        print("\nЗбереження завершено успішно.")

    except sqlite3.Error as e:
        print(f"Помилка SQLite: {e}")
    finally:
        if conn:
            conn.close()
            print("З'єднання з базою даних закрито.")

if __name__ == "__main__":
    calculate_co_occurrence(DB_FILE)
