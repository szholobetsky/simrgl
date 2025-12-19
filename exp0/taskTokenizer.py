import sqlite3
import config
from tqdm import tqdm
import re

def create_table(db_file):
    """
        Creates the MODULE and MODULE_CHANGE tables if they don't exist.

    Args:
        conn: A connection to the SQLite database.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS TOKEN (
    TOKEN_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    TOKEN TEXT UNIQUE)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS TOKEN_IDX ON TOKEN (TOKEN)
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS TASK_TOKEN_INDEX (
    TASK_ID INTEGER,
    TOKEN_ID INTEGER,
    FOREIGN KEY (TASK_ID) REFERENCES TASK(ID),  -- Assuming 'ID' is the primary key of the 'TASK' table
    FOREIGN KEY (TOKEN_ID) REFERENCES TOKEN(TOKEN_ID))
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ORDER_TOKEN_INDEX (
    PREVIOUS_TOKEN_ID INTEGER,
    CURRENT_TOKEN_ID INTEGER,
    TASK_ID INTEGER,
    FOREIGN KEY (TASK_ID) REFERENCES TASK(ID),  -- Assuming 'ID' is the primary key of the 'TASK' table
    FOREIGN KEY (CURRENT_TOKEN_ID) REFERENCES TOKEN(TOKEN_ID))
    """)    

    cursor.execute("""
    
    """)


def tokenize_and_insert(conn, cursor, task_id, title, description, comments):
    #tokens = title.split() + description.split() + comments.split()
    tokens = [token.upper() for token in re.split(r'\W+', (title or '')) + re.split(r'\W+',(description or '')) + re.split(r'\W+', (comments or ''))]
    previous_token_id = None
    for token in tokens:
        cursor.execute("SELECT TOKEN_ID FROM TOKEN WHERE TOKEN = ?", (token,))
        result = cursor.fetchone()
        if result:
            token_id = result[0]
        else:
            cursor.execute("INSERT INTO TOKEN (TOKEN) VALUES (?)", (token,))
            conn.commit()
            token_id = cursor.lastrowid
        cursor.execute("INSERT INTO TASK_TOKEN_INDEX (TASK_ID, TOKEN_ID) VALUES (?, ?)", (task_id, token_id))       
        cursor.execute("INSERT INTO ORDER_TOKEN_INDEX (PREVIOUS_TOKEN_ID, CURRENT_TOKEN_ID, TASK_ID) VALUES (?, ?, ?)", (previous_token_id, token_id, task_id))
        previous_token_id = token_id

def tokenize_all(dbFile):
    conn = sqlite3.connect(dbFile)
    cursor = conn.cursor()

    # Iterate over your TASK table and call the function for each row
    cursor.execute("SELECT ID, TITLE, DESCRIPTION, COMMENTS FROM TASK")
    tasks = cursor.fetchall()
    #count = 0
    for row in tqdm(tasks, desc="Processing tasks", unit="task", total=len(tasks)):
        #count += 1
        #if count < 15499:
        #    continue
        task_id, title, description, comments = row
        tokenize_and_insert(conn, cursor, task_id, title, description, comments)

    conn.commit()
    conn.close()

def test():
    # Connect to your SQLite database 
    dbFile=config.db_file
    create_table(dbFile)
    tokenize_all(dbFile)

if __name__ == "__main__":
    #main()
    test()
