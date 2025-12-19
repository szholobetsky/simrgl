#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commit Analysis Tool with HHI and Bradford Ranking
Analyzes Jira tasks and Git commits to calculate term concentration indices
Now supports train/test split for proper evaluation
"""

import sqlite3
import re
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import properties as props
from task_selector import TaskSelector

class CommitAnalyzer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
    
    def log(self, message: str):
        """Print log message if verbose mode is enabled"""
        if props.VERBOSE:
            print(f"[INFO] {message}")
    
    def drop_simrgl_tables(self):
        """Drop all SIMRGL_* tables if they exist"""
        self.log("Dropping existing SIMRGL_* tables...")
        
        tables_to_drop = [
            'SIMRGL_MODULE_COOCCURRENCE',
            'SIMRGL_TERM_RANK', 
            'SIMRGL_FILE_TERMS',
            'SIMRGL_MODULE_TERMS',
            'SIMRGL_TASK_TERMS',
            'SIMRGL_FILES',
            'SIMRGL_MODULE_VECTOR',
            'SIMRGL_MODULES',
            'SIMRGL_TERMS'
        ]
        
        for table in tables_to_drop:
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
                self.log(f"Dropped table {table}")
            except sqlite3.Error as e:
                self.log(f"Error dropping table {table}: {e}")
        
        self.conn.commit()
    
    def create_tables(self):
        """Create all required SIMRGL_* tables"""
        self.log("Creating SIMRGL_* tables...")
        
        # Terms table
        self.conn.execute("""
        CREATE TABLE SIMRGL_TERMS (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            TERM TEXT UNIQUE NOT NULL,
            TOTAL_COUNT INTEGER DEFAULT 0
        )
        """)
        
        # Task-Term relationships
        self.conn.execute("""
        CREATE TABLE SIMRGL_TASK_TERMS (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            TASK_ID INTEGER NOT NULL,
            TERM_ID INTEGER NOT NULL,
            COUNT INTEGER DEFAULT 1,
            FOREIGN KEY (TASK_ID) REFERENCES TASK(ID),
            FOREIGN KEY (TERM_ID) REFERENCES SIMRGL_TERMS(ID),
            UNIQUE(TASK_ID, TERM_ID)
        )
        """)
        
        # Modules table (root modules)
        self.conn.execute("""
        CREATE TABLE SIMRGL_MODULES (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            MODULE_NAME TEXT UNIQUE NOT NULL
        )
        """)
        
        # Files table
        self.conn.execute("""
        CREATE TABLE SIMRGL_FILES (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            FILE_PATH TEXT UNIQUE NOT NULL,
            MODULE_ID INTEGER NOT NULL,
            FOREIGN KEY (MODULE_ID) REFERENCES SIMRGL_MODULES(ID)
        )
        """)
        
        # Module-Term relationships
        self.conn.execute("""
        CREATE TABLE SIMRGL_MODULE_TERMS (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            MODULE_ID INTEGER NOT NULL,
            TERM_ID INTEGER NOT NULL,
            COUNT INTEGER DEFAULT 0,
            FOREIGN KEY (MODULE_ID) REFERENCES SIMRGL_MODULES(ID),
            FOREIGN KEY (TERM_ID) REFERENCES SIMRGL_TERMS(ID),
            UNIQUE(MODULE_ID, TERM_ID)
        )
        """)
        
        # File-Term relationships
        self.conn.execute("""
        CREATE TABLE SIMRGL_FILE_TERMS (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            FILE_ID INTEGER NOT NULL,
            TERM_ID INTEGER NOT NULL,
            COUNT INTEGER DEFAULT 0,
            FOREIGN KEY (FILE_ID) REFERENCES SIMRGL_FILES(ID),
            FOREIGN KEY (TERM_ID) REFERENCES SIMRGL_TERMS(ID),
            UNIQUE(FILE_ID, TERM_ID)
        )
        """)
        
        # Term ranking with HHI and Bradford
        self.conn.execute("""
        CREATE TABLE SIMRGL_TERM_RANK (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            TERM_ID INTEGER NOT NULL,
            CNT INTEGER DEFAULT 0,
            FILE_CNT INTEGER DEFAULT 0,
            ROOT_CNT INTEGER DEFAULT 0,
            HHI_FILE REAL DEFAULT 0.0,
            HHI_ROOT REAL DEFAULT 0.0,
            BRADFORD_RANK INTEGER DEFAULT 1,
            FOREIGN KEY (TERM_ID) REFERENCES SIMRGL_TERMS(ID)
        )
        """)
        
        # Module co-occurrence
        self.conn.execute("""
        CREATE TABLE SIMRGL_MODULE_COOCCURRENCE (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            L_MODULE_ID INTEGER NOT NULL,
            R_MODULE_ID INTEGER NOT NULL,
            COOCCURRENCE_COUNT INTEGER DEFAULT 0,
            FOREIGN KEY (L_MODULE_ID) REFERENCES SIMRGL_MODULES(ID),
            FOREIGN KEY (R_MODULE_ID) REFERENCES SIMRGL_MODULES(ID),
            CHECK (L_MODULE_ID < R_MODULE_ID),
            UNIQUE(L_MODULE_ID, R_MODULE_ID)
        )
        """)
        
        self.conn.commit()
        self.log("Tables created successfully")
    
    def tokenize_text(self, text: str) -> List[str]:
        """Extract words from text, keeping alphanumeric tokens"""
        if not text:
            return []
        
        # Split by whitespace and punctuation, but keep alphanumeric sequences
        tokens = re.findall(r'\b[a-zA-Z]+[a-zA-Z0-9#_]*\b|\b[a-zA-Z0-9#_]*[a-zA-Z]+\b', text.lower())
        
        filtered_tokens = []
        for token in tokens:
            # Skip if too short
            if len(token) < props.MIN_WORD_LENGTH:
                continue
                
            # Skip pure numbers if configured
            if props.IGNORE_PURE_NUMBERS and token.isdigit():
                continue
                
            # Skip pure symbols if configured  
            if props.IGNORE_PURE_SYMBOLS and not any(c.isalnum() for c in token):
                continue
                
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def extract_root_module(self, file_path: str) -> str:
        """Extract root module from file path"""
        if not file_path:
            return ""
        
        # Remove leading/trailing slashes and normalize
        file_path = file_path.strip('/')
        
        # Split by '/' and get first component
        parts = file_path.split('/')
        if len(parts) > 0 and parts[0]:
            first_part = parts[0]
            
            # If IGNORE_FILES_WITHOUT_ROOT_MODULE is True and first part contains a period,
            # treat it as a file and return empty string
            if props.IGNORE_FILES_WITHOUT_ROOT_MODULE and '.' in first_part:
                return ""
            
            return first_part
        return ""
    
    def process_tasks_and_terms(self):
        """Process tasks and extract terms (only training tasks if split is enabled)"""
        split_info = ""
        train_filter = ""
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            with TaskSelector(self.db_path) as selector:
                train_filter, _ = selector.create_train_test_filter_clause("t")
                split_info = " (training tasks only)"
        
        self.log(f"Processing tasks and extracting terms{split_info}...")
        
        # Build text source query based on configuration
        text_fields = ["t.TITLE"]
        if props.USE_DESCRIPTION:
            text_fields.append("t.DESCRIPTION")
        if props.USE_COMMENTS:
            text_fields.append("t.COMMENTS")
        
        text_concat = " || ' ' || ".join([f"COALESCE({field}, '')" for field in text_fields])
        
        query = f"""
        SELECT t.ID, t.NAME, {text_concat} as combined_text
        FROM TASK t
        WHERE t.NAME IS NOT NULL
        {train_filter}
        """
        
        cursor = self.conn.execute(query)
        tasks = cursor.fetchall()
        
        term_counts = defaultdict(int)
        task_terms = defaultdict(lambda: defaultdict(int))
        
        for task_id, task_name, combined_text in tasks:
            tokens = self.tokenize_text(combined_text)
            
            for token in tokens:
                term_counts[token] += 1
                task_terms[task_id][token] += 1
        
        # Insert terms
        self.log(f"Inserting {len(term_counts)} unique terms...")
        for term, count in term_counts.items():
            self.conn.execute(
                "INSERT OR IGNORE INTO SIMRGL_TERMS (TERM, TOTAL_COUNT) VALUES (?, ?)",
                (term, count)
            )
        
        # Get term IDs
        cursor = self.conn.execute("SELECT ID, TERM FROM SIMRGL_TERMS")
        term_id_map = {term: term_id for term_id, term in cursor.fetchall()}
        
        # Insert task-term relationships
        self.log("Inserting task-term relationships...")
        for task_id, terms in task_terms.items():
            for term, count in terms.items():
                term_id = term_id_map[term]
                self.conn.execute(
                    "INSERT OR IGNORE INTO SIMRGL_TASK_TERMS (TASK_ID, TERM_ID, COUNT) VALUES (?, ?, ?)",
                    (task_id, term_id, count)
                )
        
        self.conn.commit()
        self.log(f"Processed {len(tasks)} tasks{split_info} with {len(term_counts)} unique terms")
    
    def process_files_and_modules(self):
        """Process git commits and extract files/modules (only training tasks if split is enabled)"""
        split_info = ""
        train_filter = ""
        
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            with TaskSelector(self.db_path) as selector:
                train_tasks, _ = selector.get_train_test_split()
                train_task_names = []
                
                # Get task names for training tasks
                train_ids_str = ",".join(map(str, train_tasks))
                cursor = self.conn.execute(f"""
                SELECT NAME FROM TASK WHERE ID IN ({train_ids_str})
                """)
                train_task_names = [row[0] for row in cursor.fetchall()]
                
                if train_task_names:
                    train_task_names_str = ",".join([f"'{name}'" for name in train_task_names])
                    train_filter = f"AND r.TASK_NAME IN ({train_task_names_str})"
                
                split_info = " (training tasks only)"
        
        self.log(f"Processing files and modules from git commits{split_info}...")
        
        # Get all unique file paths from commits
        query = f"""
        SELECT DISTINCT r.PATH, r.TASK_NAME
        FROM RAWDATA r 
        WHERE r.PATH IS NOT NULL AND r.TASK_NAME IS NOT NULL
        {train_filter}
        """
        
        cursor = self.conn.execute(query)
        
        file_paths = set()
        task_files = defaultdict(set)
        
        for file_path, task_name in cursor.fetchall():
            root_module = self.extract_root_module(file_path)
            
            # Skip files without root module if configured
            if props.IGNORE_FILES_WITHOUT_ROOT_MODULE and not root_module:
                continue
                
            file_paths.add(file_path)
            task_files[task_name].add(file_path)
        
        # Extract unique modules
        modules = set()
        for file_path in file_paths:
            root_module = self.extract_root_module(file_path)
            if root_module:
                modules.add(root_module)
        
        # Insert modules
        self.log(f"Inserting {len(modules)} unique modules...")
        for module in modules:
            self.conn.execute("INSERT OR IGNORE INTO SIMRGL_MODULES (MODULE_NAME) VALUES (?)", (module,))
        
        # Get module IDs
        cursor = self.conn.execute("SELECT ID, MODULE_NAME FROM SIMRGL_MODULES")
        module_id_map = {module: module_id for module_id, module in cursor.fetchall()}
        
        # Insert files
        self.log(f"Inserting {len(file_paths)} unique files...")
        for file_path in file_paths:
            root_module = self.extract_root_module(file_path)
            if root_module and root_module in module_id_map:
                module_id = module_id_map[root_module]
                self.conn.execute(
                    "INSERT OR IGNORE INTO SIMRGL_FILES (FILE_PATH, MODULE_ID) VALUES (?, ?)",
                    (file_path, module_id)
                )
        
        self.conn.commit()
        return task_files, module_id_map
    
    def calculate_term_distributions(self, task_files: Dict[str, Set[str]]):
        """Calculate term distributions across files and modules (only training data)"""
        split_info = " (training data only)" if props.EXCLUDE_TEST_TASKS_FROM_MODEL else ""
        self.log(f"Calculating term distributions{split_info}...")
        
        # Get file and module mappings
        cursor = self.conn.execute("""
        SELECT f.ID, f.FILE_PATH, f.MODULE_ID, m.MODULE_NAME
        FROM SIMRGL_FILES f
        JOIN SIMRGL_MODULES m ON f.MODULE_ID = m.ID
        """)
        file_data = {file_path: (file_id, module_id) for file_id, file_path, module_id, module_name in cursor.fetchall()}
        
        # Get term mappings
        cursor = self.conn.execute("SELECT ID, TERM FROM SIMRGL_TERMS")
        term_id_map = {term: term_id for term_id, term in cursor.fetchall()}
        
        # Get task-term data (only for training tasks)
        train_filter = ""
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            with TaskSelector(self.db_path) as selector:
                train_filter, _ = selector.create_train_test_filter_clause("t")
        
        query = f"""
        SELECT tt.TERM_ID, t.NAME, tt.COUNT
        FROM SIMRGL_TASK_TERMS tt
        JOIN TASK t ON tt.TASK_ID = t.ID
        WHERE 1=1
        {train_filter}
        """
        
        cursor = self.conn.execute(query)
        
        # Build term distributions
        term_file_counts = defaultdict(lambda: defaultdict(int))
        term_module_counts = defaultdict(lambda: defaultdict(int))
        
        for term_id, task_name, count in cursor.fetchall():
            if task_name in task_files:
                files_in_task = task_files[task_name]
                
                for file_path in files_in_task:
                    if file_path in file_data:
                        file_id, module_id = file_data[file_path]
                        term_file_counts[term_id][file_id] += count
                        term_module_counts[term_id][module_id] += count
        
        # Insert file-term relationships
        self.log("Inserting file-term relationships...")
        for term_id, file_counts in term_file_counts.items():
            for file_id, count in file_counts.items():
                self.conn.execute(
                    "INSERT OR IGNORE INTO SIMRGL_FILE_TERMS (FILE_ID, TERM_ID, COUNT) VALUES (?, ?, ?)",
                    (file_id, term_id, count)
                )
        
        # Insert module-term relationships
        self.log("Inserting module-term relationships...")
        for term_id, module_counts in term_module_counts.items():
            for module_id, count in module_counts.items():
                self.conn.execute(
                    "INSERT OR IGNORE INTO SIMRGL_MODULE_TERMS (MODULE_ID, TERM_ID, COUNT) VALUES (?, ?, ?)",
                    (module_id, term_id, count)
                )
        
        self.conn.commit()
        return term_file_counts, term_module_counts
    
    def calculate_hhi(self, counts: Dict[int, int]) -> float:
        """Calculate Herfindahl-Hirschman Index"""
        if not counts:
            return 0.0
        
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        hhi = sum((count / total) ** 2 for count in counts.values())
        return hhi
    
    def calculate_bradford_ranks(self) -> Dict[int, int]:
        """Calculate Bradford ranking zones for terms"""
        self.log("Calculating Bradford ranking zones...")
        
        # Get term frequencies
        cursor = self.conn.execute("SELECT ID, TOTAL_COUNT FROM SIMRGL_TERMS ORDER BY TOTAL_COUNT DESC")
        terms_by_freq = cursor.fetchall()
        
        if not terms_by_freq:
            return {}
        
        # Calculate total frequency
        total_freq = sum(count for _, count in terms_by_freq)
        
        # Calculate zone boundaries using Bradford's law
        # Each zone should contain approximately the same total frequency
        zone_target_freq = total_freq / props.BRADFORD_ZONES
        
        bradford_ranks = {}
        current_zone = 1
        current_zone_freq = 0
        
        for term_id, freq in terms_by_freq:
            bradford_ranks[term_id] = current_zone
            current_zone_freq += freq
            
            # Move to next zone when current zone target is reached
            if current_zone_freq >= zone_target_freq and current_zone < props.BRADFORD_ZONES:
                current_zone += 1
                current_zone_freq = 0
        
        self.log(f"Assigned Bradford ranks to {len(bradford_ranks)} terms in {props.BRADFORD_ZONES} zones")
        return bradford_ranks
    
    def calculate_term_rankings(self, term_file_counts: Dict[int, Dict[int, int]], 
                               term_module_counts: Dict[int, Dict[int, int]]):
        """Calculate final term rankings with HHI and Bradford"""
        self.log("Calculating term rankings...")
        
        bradford_ranks = self.calculate_bradford_ranks()
        
        # Get all terms
        cursor = self.conn.execute("SELECT ID, TOTAL_COUNT FROM SIMRGL_TERMS")
        all_terms = cursor.fetchall()
        
        for term_id, total_count in all_terms:
            file_counts = term_file_counts.get(term_id, {})
            module_counts = term_module_counts.get(term_id, {})
            
            hhi_file = self.calculate_hhi(file_counts)
            hhi_module = self.calculate_hhi(module_counts)
            
            file_cnt = len(file_counts)
            root_cnt = len(module_counts)
            bradford_rank = bradford_ranks.get(term_id, props.BRADFORD_ZONES)
            
            self.conn.execute("""
            INSERT INTO SIMRGL_TERM_RANK 
            (TERM_ID, CNT, FILE_CNT, ROOT_CNT, HHI_FILE, HHI_ROOT, BRADFORD_RANK)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (term_id, total_count, file_cnt, root_cnt, hhi_file, hhi_module, bradford_rank))
        
        self.conn.commit()
        self.log(f"Calculated rankings for {len(all_terms)} terms")
    
    def calculate_module_cooccurrence(self, task_files: Dict[str, Set[str]]):
        """Calculate module co-occurrence in tasks"""
        self.log("Calculating module co-occurrence...")
        
        # Get module mappings
        cursor = self.conn.execute("""
        SELECT f.FILE_PATH, f.MODULE_ID
        FROM SIMRGL_FILES f
        """)
        file_to_module = {file_path: module_id for file_path, module_id in cursor.fetchall()}
        
        # Count module pairs per task
        module_pairs = defaultdict(int)
        
        for task_name, files in task_files.items():
            # Get unique modules in this task
            modules_in_task = set()
            for file_path in files:
                if file_path in file_to_module:
                    modules_in_task.add(file_to_module[file_path])
            
            # Generate all pairs of modules (L_MODULE_ID < R_MODULE_ID)
            modules_list = sorted(list(modules_in_task))
            for i in range(len(modules_list)):
                for j in range(i + 1, len(modules_list)):
                    left_id, right_id = modules_list[i], modules_list[j]
                    if left_id < right_id:
                        module_pairs[(left_id, right_id)] += 1
        
        # Insert co-occurrence data
        self.log(f"Inserting {len(module_pairs)} module co-occurrence pairs...")
        for (left_id, right_id), count in module_pairs.items():
            self.conn.execute(
                "INSERT OR IGNORE INTO SIMRGL_MODULE_COOCCURRENCE (L_MODULE_ID, R_MODULE_ID, COOCCURRENCE_COUNT) VALUES (?, ?, ?)",
                (left_id, right_id, count)
            )
        
        self.conn.commit()
        self.log(f"Calculated co-occurrence for {len(module_pairs)} module pairs")
    
    def print_summary_statistics(self):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            print("(Training data only - test tasks excluded from model)")
        print("="*60)
        
        # Terms
        cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_TERMS")
        terms_count = cursor.fetchone()[0]
        print(f"Total unique terms: {terms_count}")
        
        # Modules  
        cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_MODULES")
        modules_count = cursor.fetchone()[0]
        print(f"Total modules: {modules_count}")
        
        # Files
        cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_FILES")
        files_count = cursor.fetchone()[0]
        print(f"Total files: {files_count}")
        
        # Module pairs
        cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_MODULE_COOCCURRENCE")
        pairs_count = cursor.fetchone()[0]
        print(f"Module co-occurrence pairs: {pairs_count}")
        
        # Bradford zones
        print(f"\nBradford zones distribution:")
        for zone in range(1, props.BRADFORD_ZONES + 1):
            cursor = self.conn.execute("SELECT COUNT(*) FROM SIMRGL_TERM_RANK WHERE BRADFORD_RANK = ?", (zone,))
            count = cursor.fetchone()[0]
            print(f"  Zone {zone}: {count} terms")
        
        # Top terms by HHI
        print(f"\nTop 10 terms by HHI (module concentration):")
        cursor = self.conn.execute("""
        SELECT t.TERM, tr.HHI_ROOT, tr.ROOT_CNT, tr.CNT
        FROM SIMRGL_TERM_RANK tr
        JOIN SIMRGL_TERMS t ON tr.TERM_ID = t.ID
        ORDER BY tr.HHI_ROOT DESC
        LIMIT 10
        """)
        
        for term, hhi_root, root_cnt, cnt in cursor.fetchall():
            print(f"  {term}: HHI={hhi_root:.3f}, modules={root_cnt}, total={cnt}")
        
        # Show train/test split info
        if props.EXCLUDE_TEST_TASKS_FROM_MODEL:
            try:
                with TaskSelector(self.db_path) as selector:
                    selector.print_split_summary()
            except Exception as e:
                self.log(f"Error showing split summary: {e}")
        
        print("="*60)
    
    def run_analysis(self):
        """Run the complete analysis"""
        self.log("Starting commit analysis...")
        
        # Step 1: Setup
        self.drop_simrgl_tables()
        self.create_tables()
        
        # Step 2: Process tasks and extract terms
        self.process_tasks_and_terms()
        
        # Step 3: Process files and modules
        task_files, module_id_map = self.process_files_and_modules()
        
        # Step 4: Calculate term distributions
        term_file_counts, term_module_counts = self.calculate_term_distributions(task_files)
        
        # Step 5: Calculate rankings
        self.calculate_term_rankings(term_file_counts, term_module_counts)
        
        # Step 6: Calculate module co-occurrence
        self.calculate_module_cooccurrence(task_files)
        
        # Step 7: Print results
        self.print_summary_statistics()
        
        self.log("Analysis completed successfully!")

def main():
    """Main entry point"""
    try:
        with CommitAnalyzer(props.DATABASE_PATH) as analyzer:
            analyzer.run_analysis()
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()