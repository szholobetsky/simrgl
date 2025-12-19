import sqlite3
import os

# --- Налаштування ---
DB_FILE = "flink_mobile.db"  # Назва вашого файлу бази даних
RAWDATA_TABLE = "RAWDATA"
MODULE_TABLE = "MODULE"
MODULE_TASK_TABLE = "MODULE_TASK"

def create_and_populate_modules(db_file):
    """
    Обробляє таблицю RAWDATA, створює та заповнює таблиці MODULE і MODULE_TASK.
    """
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        print("Підключено до бази даних.")

        # --- Створення таблиць ---
        print("Створення таблиць...")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {MODULE_TABLE} (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                PARENT_ID INTEGER,
                NAME TEXT NOT NULL,
                LVL INTEGER NOT NULL,
                FILE INTEGER NOT NULL,
                CNT INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (PARENT_ID) REFERENCES {MODULE_TABLE}(ID),
                UNIQUE(PARENT_ID, NAME)
            );
        """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {MODULE_TASK_TABLE} (
                TASK_NAME TEXT NOT NULL,
                MODULE_ID INTEGER NOT NULL,
                CNT INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (TASK_NAME, MODULE_ID),
                FOREIGN KEY (MODULE_ID) REFERENCES {MODULE_TABLE}(ID)
            );
        """)
        conn.commit()
        print("Таблиці створено (якщо не існували).")

        # --- Отримання даних з таблиці RAWDATA ---
        print(f"Зчитування даних з таблиці {RAWDATA_TABLE}...")
        cursor.execute(f"SELECT TASK_NAME, PATH FROM {RAWDATA_TABLE}")
        raw_data = cursor.fetchall()

        if not raw_data:
            print(f"Таблиця {RAWDATA_TABLE} порожня. Завершення.")
            return
        
        # --- Обробка та заповнення таблиць ---
        for task_name, path in raw_data:
            # --- ВИПРАВЛЕННЯ ---
            # Пропускаємо рядки, де task_name або path порожні
            if not task_name or not path:
                print(f"Пропущено рядок через порожні дані: TASK_NAME='{task_name}', PATH='{path}'")
                continue
            # --- КІНЕЦЬ ВИПРАВЛЕННЯ ---

            # Нормалізуємо шлях і розбиваємо на компоненти
            parts = [p for p in path.strip('/').split('/') if p]
            parent_id = None

            for i, part_name in enumerate(parts):
                is_file = 1 if i == len(parts) - 1 else 0
                lvl = i

                # Знаходимо або створюємо модуль
                cursor.execute(f"SELECT ID FROM {MODULE_TABLE} WHERE NAME = ? AND PARENT_ID {'IS' if parent_id is None else '='} ?", (part_name, parent_id))
                module_row = cursor.fetchone()

                if module_row:
                    module_id = module_row[0]
                    # Оновлюємо лічильник CNT для існуючого модуля
                    cursor.execute(f"UPDATE {MODULE_TABLE} SET CNT = CNT + 1 WHERE ID = ?", (module_id,))
                else:
                    # Вставляємо новий модуль
                    cursor.execute(f"INSERT INTO {MODULE_TABLE} (PARENT_ID, NAME, LVL, FILE, CNT) VALUES (?, ?, ?, ?, 1)", (parent_id, part_name, lvl, is_file))
                    module_id = cursor.lastrowid
                
                # Оновлюємо лічильник CNT у таблиці MODULE_TASK
                cursor.execute(f"INSERT INTO {MODULE_TASK_TABLE} (TASK_NAME, MODULE_ID, CNT) VALUES (?, ?, 1) ON CONFLICT(TASK_NAME, MODULE_ID) DO UPDATE SET CNT = CNT + 1", (task_name, module_id))
                
                # Оновлюємо parent_id для наступної ітерації
                parent_id = module_id
        
        conn.commit()
        print("Дані успішно заповнено.")

    except sqlite3.Error as e:
        print(f"Помилка SQLite: {e}")
    finally:
        if conn:
            conn.close()
            print("З'єднання з базою даних закрито.")

if __name__ == "__main__":
    create_and_populate_modules(DB_FILE)
