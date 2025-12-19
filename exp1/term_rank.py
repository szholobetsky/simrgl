import sqlite3
from tqdm import tqdm
from collections import defaultdict

# --- Налаштування ---
DB_FILE = "data/sonar.db"
TERM_TABLE = "TITLE_TERM"
TASK_TERM_TABLE = "TITLE_TASK_TERM"
MODULE_TABLE = "MODULE"
MODULE_TASK_TABLE = "MODULE_TASK"
TERM_RANK_TABLE = "TERM_RANK"

def get_path_components(conn, module_id, cache):
    """
    Рекурсивно отримує компоненти шляху для модуля, використовуючи кеш.
    """
    if module_id in cache:
        return cache[module_id]

    cursor = conn.cursor()
    path_components = []
    current_id = module_id
    
    while current_id is not None:
        cursor.execute(f"SELECT PARENT_ID, NAME FROM {MODULE_TABLE} WHERE ID = ?", (current_id,))
        row = cursor.fetchone()
        if not row:
            break
        
        parent_id, name = row
        path_components.insert(0, name)
        current_id = parent_id
        
    cache[module_id] = path_components
    return path_components

def find_common_ancestor_lvl(paths):
    """
    Знаходить рівень найглибшого спільного предка.
    """
    if not paths:
        return -1
    
    min_len = min(len(p) for p in paths)
    
    common_prefix_len = 0
    for i in range(min_len):
        current_component = paths[0][i]
        if all(p[i] == current_component for p in paths):
            common_prefix_len += 1
        else:
            break
    
    return common_prefix_len - 1

def calculate_term_metrics(db_file):
    """
    Розраховує та заповнює таблицю TERM_RANK з додатковими метриками.
    """
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        print("Підключено до бази даних.")
        
        # ОНОВЛЕННЯ: Створення таблиці TERM_RANK з новими полями
        print("Створення/оновлення таблиці TERM_RANK...")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TERM_RANK_TABLE} (
                TERM_ID INTEGER PRIMARY KEY,
                RANK INTEGER NOT NULL,
                TASK_CNT INTEGER NOT NULL DEFAULT 0,
                FILE_CNT INTEGER NOT NULL DEFAULT 0,
                ROOT_CNT INTEGER NOT NULL DEFAULT 0,
                HHI_FILE REAL NOT NULL DEFAULT 0.0,
                HHI_ROOT REAL NOT NULL DEFAULT 0.0,
                COMPOSITE_INDEX REAL NOT NULL DEFAULT 0.0,
                FOREIGN KEY (TERM_ID) REFERENCES {TERM_TABLE}(ID)
            );
        """)
        conn.commit()
        
        cursor.execute(f"SELECT ID FROM {TERM_TABLE}")
        term_ids = [row[0] for row in cursor.fetchall()]
        
        if not term_ids:
            print("Немає термінів для ранжування.")
            return

        module_path_cache = {}
        
        print(f"Починаю розрахунок метрик для {len(term_ids)} термінів...")
        with tqdm(total=len(term_ids), unit="термін") as pbar:
            for term_id in term_ids:
                cursor.execute(f"SELECT DISTINCT TASK_NAME FROM {TASK_TERM_TABLE} WHERE TERM_ID = ?", (term_id,))
                task_names = [row[0] for row in cursor.fetchall()]
                
                task_cnt = len(task_names)
                rank, file_cnt, root_cnt = 0, 0, 0
                hhi_file, hhi_root, composite_index = 0.0, 0.0, 0.0
                
                if task_cnt > 0:
                    modules = set()
                    for task_name in task_names:
                        cursor.execute(f"SELECT MODULE_ID FROM {MODULE_TASK_TABLE} WHERE TASK_NAME = ?", (task_name,))
                        for row in cursor.fetchall():
                            modules.add(row[0])
                    
                    if modules:
                        # Збір даних для розрахунків
                        module_appearances = defaultdict(int)
                        file_appearances = defaultdict(int)
                        root_appearances = defaultdict(int)
                        total_appearances = 0

                        for task_name in task_names:
                            # Отримуємо кількість появ терміна в назві таска
                            cursor.execute(f"SELECT CNT FROM TITLE_TASK_TERM_AGG WHERE TASK_NAME = ? AND TERM_ID = ?", (task_name, term_id))
                            term_in_task_cnt = cursor.fetchone()[0]

                            # Отримуємо модулі, пов'язані з цим таском
                            cursor.execute(f"SELECT MODULE_ID FROM {MODULE_TASK_TABLE} WHERE TASK_NAME = ?", (task_name,))
                            module_ids_in_task = [row[0] for row in cursor.fetchall()]

                            for m_id in module_ids_in_task:
                                module_appearances[m_id] += term_in_task_cnt
                                total_appearances += term_in_task_cnt
                        
                        file_cnt_unique = 0
                        root_cnt_unique = 0
                        all_paths = []
                        
                        # Розрахунок HHI_file та HHI_root
                        if total_appearances > 0:
                            hhi_file = 0.0
                            hhi_root = 0.0
                            
                            for m_id, cnt in module_appearances.items():
                                cursor.execute(f"SELECT FILE, NAME FROM {MODULE_TABLE} WHERE ID = ?", (m_id,))
                                is_file, module_name = cursor.fetchone()
                                
                                path_components = get_path_components(conn, m_id, module_path_cache)
                                all_paths.append(path_components)

                                if is_file:
                                    file_cnt_unique += 1
                                    share = cnt / total_appearances
                                    hhi_file += share ** 2
                                    file_appearances[m_id] += cnt # Для підрахунку загальної кількості появ у файлах

                                if path_components:
                                    root_name = path_components[0]
                                    root_appearances[root_name] += cnt
                            
                            root_cnt_unique = len(root_appearances)
                            
                            # Розрахунок HHI_root на основі загальної кількості появ у кореневих модулях
                            for root_name, cnt in root_appearances.items():
                                share = cnt / total_appearances
                                hhi_root += share ** 2
                                
                            # Розрахунок композитного індексу
                            composite_index = hhi_file * hhi_root
                            
                            # Розрахунок рангу
                            common_ancestor_lvl = find_common_ancestor_lvl(all_paths)
                            max_lvl = max((len(p) - 1) for p in all_paths)
                            rank = max_lvl - common_ancestor_lvl
                            file_cnt = file_cnt_unique
                            root_cnt = root_cnt_unique

                # Вставляємо всі параметри в таблицю
                cursor.execute(f"INSERT OR REPLACE INTO {TERM_RANK_TABLE} (TERM_ID, RANK, TASK_CNT, FILE_CNT, ROOT_CNT, HHI_FILE, HHI_ROOT, COMPOSITE_INDEX) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (term_id, rank, task_cnt, file_cnt, root_cnt, hhi_file, hhi_root, composite_index))
                pbar.update(1)
        
        conn.commit()
        print("\nРозрахунок метрик завершено успішно.")

    except sqlite3.Error as e:
        print(f"Помилка SQLite: {e}")
    finally:
        if conn:
            conn.close()
            print("З'єднання з базою даних закрито.")

if __name__ == "__main__":
    calculate_term_metrics(DB_FILE)
