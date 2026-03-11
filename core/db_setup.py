import sqlite3
import sqlite_vec
import os


class DatabaseManager:
    def __init__(self, db_path="db/data.db"):
        self.db_path = db_path

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = self._connect()
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        return conn

    def init_database(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT UNIQUE NOT NULL,
            file_type TEXT,
            created_at DATETIME,
            modified_at DATETIME,
            file_size_kb REAL,
            page_count INTEGER,     -- strony (pdf), slajdy (pptx/ppt), linie (pliki tekstowe)
            topic TEXT,             -- np. 'Fizyka', 'Faktury'
            summary TEXT            -- krótkie podsumowanie z LLM
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            chunk_index INTEGER,
            content TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        """)

        cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_vec USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[768]
        );
        """)

        existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(documents)").fetchall()]
        if 'page_count' not in existing_cols:
            cursor.execute("ALTER TABLE documents ADD COLUMN page_count INTEGER;")

        self.conn.commit()

    def select_data(self):
        cursor = self.conn.cursor()

        cursor.execute("SELECT filename, topic, summary, page_count FROM documents;")

        rows = cursor.fetchall()

        for row in rows:
            file_data = dict(row)
            print(f"{file_data['filename']} ({file_data['topic']})\n{file_data['summary']}\nPage count: {file_data['page_count']}\n")

    def delete_data(self):
        cursor = self.conn.cursor()

        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM document_chunks")
        cursor.execute("DELETE FROM document_chunks_vec")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('documents', 'document_chunks');")

        self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = DatabaseManager()
    # db.delete_data()
    # db.init_database()
    db.select_data()
    
    db.close()