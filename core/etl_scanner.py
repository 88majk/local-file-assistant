import os
import json
from datetime import datetime
from markitdown import MarkItDown
import ollama
from db_setup import DatabaseManager
from concurrent.futures import ThreadPoolExecutor, as_completed

TEXT_EXTENSIONS = {'txt', 'md', 'py', 'c', 'cpp', 'h', 'js', 'ts', 'json', 'xml', 'csv', 'html', 'css', 'java', 'cs', 'go', 'rs'}


class ETLScanner:
    def __init__(self, folder_path="data", max_workers=3):
        self.folder_path = folder_path
        self.db = DatabaseManager(db_path="db/data.db")
        self.db.init_database()
        self.max_workers = max_workers

        self.LLM_MODEL = "llama3.2:1b"
        self.EMBED_MODEL = "nomic-embed-text"

    def get_file_metadata(self, filepath):
        stat = os.stat(filepath)
        return {
            "size_kb": round(stat.st_size / 1024, 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            "file_type": filepath.split('.')[-1].lower()
        }

    def get_page_count(self, filepath, file_type, full_text):
        try:
            if file_type == 'pdf':
                from pypdf import PdfReader
                return len(PdfReader(filepath).pages)
            elif file_type in ('pptx', 'ppt'):
                from pptx import Presentation
                return len(Presentation(filepath).slides)
            elif file_type in TEXT_EXTENSIONS:
                return full_text.count('\n') + 1 if full_text else None
        except Exception as e:
            print(f"Nie udało się odczytać page_count: {e}")
        return None

    def analyze_with_llm(self, text):
        prompt = f"""
            Przeanalizuj poniższy tekst i zwróć wynik WYŁĄCZNIE jako poprawny obiekt JSON.
            Wymagane pola to:
            "topic": krótka kategoria/temat (max 2 słowa, np. "Fizyka", "Faktury", "Umowa")
            "summary": jednozdaniowe streszczenie o czym jest ten dokument.
    
            Tekst:
            {text[:2000]}
        """

        try:
            response = ollama.chat(model=self.LLM_MODEL, messages=[
                {'role': 'user', 'content': prompt}
            ], format='json') 

            result = json.loads(response['message']['content'])
            return result.get('topic', 'Inne'), result.get('summary', 'Brak podsumowania')
        except Exception as e:
            return "Nieznany", "Błąd generowania podsumowania."

    def chunk_text(self, text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def _process_file_data(self, filename, filepath):
        meta = self.get_file_metadata(filepath)

        try:
            md_result = MarkItDown().convert(filepath)
            full_text = md_result.text_content
        except Exception as e:
            return None

        topic, summary = self.analyze_with_llm(full_text)
        page_count = self.get_page_count(filepath, meta['file_type'], full_text)
        chunks = self.chunk_text(full_text)

        try:
            embed_response = ollama.embed(model=self.EMBED_MODEL, input=chunks)
            embeddings = embed_response['embeddings']
        except Exception as e:
            return None

        return {
            'filename': filename,
            'filepath': filepath,
            'meta': meta,
            'topic': topic,
            'summary': summary,
            'page_count': page_count,
            'chunks': chunks,
            'embeddings': embeddings,
        }

    def process_folder(self):
        if not os.path.exists(self.folder_path):
            return

        files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]

        cursor = self.db.conn.cursor()

        to_process = []
        for filename in files:
            filepath = os.path.join(self.folder_path, filename)
            cursor.execute("SELECT id FROM documents WHERE filepath = ?", (filepath,))

        if not to_process:
            self.db.close()
            return

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_file_data, fn, fp): fn
                for fn, fp in to_process
            }
            for future in as_completed(futures):
                data = future.result()
                if data:
                    results.append(data)

        for data in results:
            cursor.execute("""
                INSERT INTO documents (filename, filepath, file_type, created_at, modified_at, file_size_kb, page_count, topic, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['filename'], data['filepath'],
                data['meta']['file_type'], data['meta']['created_at'],
                data['meta']['modified_at'], data['meta']['size_kb'],
                data['page_count'], data['topic'], data['summary']
            ))
            doc_id = cursor.lastrowid

            cursor.executemany("""
                INSERT INTO document_chunks (document_id, chunk_index, content) VALUES (?, ?, ?)
            """, [(doc_id, i, chunk) for i, chunk in enumerate(data['chunks'])])

            cursor.execute(
                "SELECT id FROM document_chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc_id,)
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]

            cursor.executemany("""
                INSERT INTO document_chunks_vec (chunk_id, embedding) VALUES (?, ?)
            """, [(chunk_id, json.dumps(emb)) for chunk_id, emb in zip(chunk_ids, data['embeddings'])])

            self.db.conn.commit()

        self.db.close()


if __name__ == "__main__":
    scanner = ETLScanner()
    scanner.process_folder()