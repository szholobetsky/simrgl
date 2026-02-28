import sqlite3
from collections import defaultdict

# Connect to the database
conn = sqlite3.connect('your_database.db')  # Replace with your database name
cursor = conn.cursor()

# Step 1: Fetch token counts (how many tasks each token appears in)
token_counts = {}
cursor.execute("""
    SELECT TOKEN_ID, COUNT(DISTINCT TASK_ID) AS total_count
    FROM TASK_TOKEN_INDEX
    GROUP BY TOKEN_ID
""")
for row in cursor.fetchall():
    token_id = row[0]
    count = row[1]
    token_counts[token_id] = count

# Step 2: Fetch all tasks and their associated tokens
task_tokens = {}
cursor.execute("""
    SELECT TASK_ID, GROUP_CONCAT(TOKEN_ID) AS tokens
    FROM TASK_TOKEN_INDEX
    GROUP BY TASK_ID
""")
for row in cursor.fetchall():
    task_id = row[0]
    tokens_str = row[1]
    tokens = list(map(int, tokens_str.split(',')))  # Convert to list of token IDs
    task_tokens[task_id] = tokens

# Step 3: Compute co-occurrence counts for all unordered pairs
co_occurrence = defaultdict(int)
for task_id, tokens in task_tokens.items():
    # Generate all unordered pairs (min, max) in this task
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            a, b = sorted([tokens[i], tokens[j]])
            co_occurrence[(a, b)] += 1

# Step 4: Sort pairs by co-occurrence count in descending order
sorted_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)

# Step 5: Select non-overlapping pairs where co-occurrence > both individual counts
selected_pairs = []
used_tokens = set()

for pair, count in sorted_pairs:
    token_a, token_b = pair
    if token_a not in used_tokens and token_b not in used_tokens:
        # Ensure co-occurrence is greater than individual counts of both tokens
        if count > token_counts.get(token_a, 0) and count > token_counts.get(token_b, 0):
            selected_pairs.append((token_a, token_b))
            used_tokens.add(token_a)
            used_tokens.add(token_b)

# Step 6: Output the selected pairs
print("Selected Pairs (Co-occurrence > individual counts, non-overlapping):")
for pair in selected_pairs:
    print(f"Pair: {pair} (Co-occurrence: {co_occurrence[pair]})")
    print(f"Token A count: {token_counts.get(pair[0], 0)}")
    print(f"Token B count: {token_counts.get(pair[1], 0)}")
    print("---")

conn.close()