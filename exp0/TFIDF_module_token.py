import sqlite3
import pandas as pd
from tqdm import tqdm
import config
import math

def calculate_tfidf_in_batches(conn, batch_size=1000):
    # Get all unique tokens from the database
    # query = "SELECT DISTINCT TOKEN_ID FROM TOKEN"
    query = "SELECT TOKEN_ID FROM TASK_TOKEN_INDEX tti WHERE tti.TOKEN_ID > 170 group by TOKEN_ID HAVING COUNT(*) > 100 "
    token_ids = pd.read_sql_query(query, conn)['TOKEN_ID'].tolist()

    total_modules = pd.read_sql_query("SELECT COUNT(*) FROM MODULE WHERE LEVEL = 0", conn).iloc[0, 0]

    for token_id in tqdm(token_ids):
        # Calculate global token statistics
        #global_token_count = pd.read_sql_query(
        #    "SELECT COUNT(*) FROM MODULE_CHANGE mc JOIN (SELECT * FROM MODULE WHERE LEVEL = 0) m ON mc.MODULE_ID = m.ID JOIN TASK_TOKEN_INDEX tti ON mc.TASK_ID = tti.TASK_ID WHERE tti.TOKEN_ID = ?",
        #    conn,
        #    params=(token_id,)
        #).iloc[0, 0]

        # Calculate TF-IDF for the token in each module
        query = """
            SELECT mc.MODULE_ID as MODULE_ID, sum(token_in_task) AS token_count -- token_in_module
            FROM (SELECT TASK_ID, COUNT(*) token_in_task FROM TASK_TOKEN_INDEX WHERE TOKEN_ID = ? group by TASK_ID) tti 
            JOIN (select distinct module_id, task_id 
                  FROM MODULE_CHANGE mmc 
                  JOIN (SELECT ID, NAME FROM MODULE WHERE LEVEL = 0) m ON mmc.MODULE_ID = m.ID ) mc ON mc.TASK_ID = tti.TASK_ID            
            GROUP BY mc.MODULE_ID
        """
        module_token_counts = pd.read_sql_query(query, conn, params=(token_id,))

        global_token_count = len(module_token_counts)

        for _, row in tqdm(module_token_counts.iterrows(), total=len(module_token_counts), desc=f"Processing modules in token {token_id}"):
            module_id = int(row['MODULE_ID'])
            token_count_in_module = int(row['token_count'])

            # Calculate the total number of tokens in the module
            total_tokens_in_module_query = """
                SELECT COUNT(*) AS total_tokens
                FROM TASK_TOKEN_INDEX tti
                JOIN (SELECT distinct TASK_ID 
                      FROM MODULE_CHANGE mc 
                      WHERE mc.MODULE_ID = ?
                ) mc ON tti.TASK_ID = mc.TASK_ID
                
            """
            total_tokens_in_module = pd.read_sql_query(total_tokens_in_module_query, conn, params=(module_id,)).iloc[0, 0]

            tf = token_count_in_module / total_tokens_in_module if total_tokens_in_module != 0 else 0  # Calculate TF
            idf = math.log(total_modules / global_token_count) if (total_modules > global_token_count) and (global_token_count != 0) else 0 # Calculate IDF
            tfidf = tf * idf

            # Insert TF-IDF and components into the database
            insert_query = """
                INSERT INTO TFIDF_MODULE_TOKEN (MODULE_ID, TOKEN_ID, TFIDF, TF, IDF)
                VALUES (?, ?, ?, ?, ?)
            """
            conn.execute(insert_query, (module_id, token_id, tfidf, tf, idf))

        conn.commit()

# Connect to the database
conn = sqlite3.connect(config.db_file)

# Calculate and insert TF-IDF scores
calculate_tfidf_in_batches(conn)

conn.close()
# Connect to the database
