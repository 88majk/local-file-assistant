import sqlite3
import sqlite_vec
import ollama
from datetime import datetime
import re
import json

POLISH_STOPWORDS = {
    'i', 'oraz', 'a', 'w', 'z', 'ze', 'na', 'do', 'od', 'u', 'o', 'po', 'za', 'dla',
    'co', 'to', 'ten', 'ta', 'te', 'jak', 'jakie', 'jaki', 'ktorych', 'ktore', 'ktora',
    'mam', 'miec', 'czy', 'nie', 'jest', 'sa', 'byc', 'tematu', 'plik', 'pliki', 'dokument', 'dokumenty'
}


class TextToSQL:
    def __init__(self, db_path="db/data.db"):
        self.db_path = db_path
        self.llm_model = "qwen2.5-coder:7b"
        self.embed_model = "nomic-embed-text"

        self.schema = """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            filepath TEXT UNIQUE NOT NULL,
            file_type TEXT,
            created_at DATETIME,
            modified_at DATETIME,
            file_size_kb REAL,
            page_count INTEGER,  -- liczba stron (pdf), slajdów (pptx/ppt) lub linii (pliki tekstowe), NULL dla pozostałych
            topic TEXT,
            summary TEXT
        );
        """

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def embed_query(self, query_text):
        response = ollama.embed(model=self.embed_model, input=[query_text])
        return response['embeddings'][0]

    def _keywords_from_query(self, query_text):
        tokens = re.findall(r"[a-zA-Z0-9ąćęłńóśźż]+", query_text.lower())
        return [t for t in tokens if len(t) >= 3 and t not in POLISH_STOPWORDS]

    def _keyword_overlap_score(self, keywords, *texts):
        if not keywords:
            return 0.0
        haystack = " ".join([t for t in texts if t]).lower()
        matched = sum(1 for kw in keywords if kw in haystack)
        return matched / len(keywords)

    def search_semantic_chunks(self, query_text, top_k=15, file_type=None, min_page_count=None):
        """Zwraja top-k chunków wg odległości wektorowej oraz metadanych dokumentu."""
        conn = self._connect()
        cursor = conn.cursor()

        query_embedding = json.dumps(self.embed_query(query_text))

        where_clauses = []
        candidate_k = max(top_k * 4, 40)
        params = [query_embedding, candidate_k]
        if file_type:
            where_clauses.append("d.file_type = ?")
            params.append(file_type.lower())
        if min_page_count is not None:
            where_clauses.append("COALESCE(d.page_count, 0) >= ?")
            params.append(min_page_count)

        filter_sql = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT
                d.id AS document_id,
                d.filename,
                d.file_type,
                d.page_count,
                d.topic,
                d.summary,
                c.id AS chunk_id,
                c.chunk_index,
                c.content,
                v.distance
            FROM document_chunks_vec v
            JOIN document_chunks c ON c.id = v.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE v.embedding MATCH ?
              AND v.k = ?
              {filter_sql}
            ORDER BY v.distance ASC
        """

        try:
            cursor.execute(sql, params)
            rows = [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            fallback_sql = f"""
                SELECT
                    d.id AS document_id,
                    d.filename,
                    d.file_type,
                    d.page_count,
                    d.topic,
                    d.summary,
                    c.id AS chunk_id,
                    c.chunk_index,
                    c.content,
                    vec_distance_l2(v.embedding, ?) AS distance
                FROM document_chunks_vec v
                JOIN document_chunks c ON c.id = v.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE 1=1
                  {filter_sql}
                ORDER BY distance ASC
                LIMIT ?
            """
            fallback_params = [query_embedding] + params[2:] + [candidate_k]
            cursor.execute(fallback_sql, fallback_params)
            rows = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return rows

    def rank_documents(self, chunk_hits, query_text, top_docs=5, min_overlap=0.2):
        keywords = self._keywords_from_query(query_text)
        by_doc = {}
        for idx, hit in enumerate(chunk_hits, start=1):
            doc_id = hit['document_id']
            if doc_id not in by_doc:
                by_doc[doc_id] = {
                    'document_id': doc_id,
                    'filename': hit['filename'],
                    'file_type': hit['file_type'],
                    'page_count': hit['page_count'],
                    'topic': hit['topic'],
                    'summary': hit['summary'],
                    'best_distance': hit['distance'],
                    'chunk_hits': 0,
                    'rrf_score': 0.0,
                    'keyword_overlap': 0.0,
                    'evidence': []
                }

            item = by_doc[doc_id]
            item['chunk_hits'] += 1
            item['best_distance'] = min(item['best_distance'], hit['distance'])
            item['rrf_score'] += 1.0 / (50 + idx)
            if len(item['evidence']) < 3:
                item['evidence'].append({
                    'chunk_index': hit['chunk_index'],
                    'distance': hit['distance'],
                    'snippet': hit['content'][:280].replace('\n', ' ')
                })

        ranked = list(by_doc.values())
        for item in ranked:
            evidence_text = " ".join([e['snippet'] for e in item['evidence']])
            item['keyword_overlap'] = self._keyword_overlap_score(
                keywords,
                item.get('topic', ''),
                item.get('summary', ''),
                evidence_text,
            )

            semantic_score = 1.0 / (1.0 + item['best_distance'])
            density_score = min(item['chunk_hits'], 8) * 0.05
            lexical_score = item['keyword_overlap'] * 0.45
            rrf_score = item['rrf_score'] * 2.0
            item['score'] = semantic_score + density_score + lexical_score + rrf_score

        ranked.sort(key=lambda x: x['score'], reverse=True)

        if keywords:
            filtered = [d for d in ranked if d['keyword_overlap'] >= min_overlap]
            if filtered:
                ranked = filtered

        return ranked[:top_docs]

    def ask_semantic(self, user_query, top_k_chunks=20, top_docs=5, file_type=None, min_page_count=None):
        print(f"\nUżytkownik: {user_query}")

        chunk_hits = self.search_semantic_chunks(
            query_text=user_query,
            top_k=top_k_chunks,
            file_type=file_type,
            min_page_count=min_page_count,
        )

        if not chunk_hits:
            return []

        docs = self.rank_documents(chunk_hits, query_text=user_query, top_docs=top_docs)
        print(f"✅ Top {len(docs)} dokumentów (po semantyce):")
        for i, d in enumerate(docs, start=1):
            print(
                f"   {i}. {d['filename']} | typ: {d['file_type']} | "
                f"page_count: {d['page_count']} | score: {d['score']:.4f} | overlap: {d['keyword_overlap']:.2f}"
            )

        return docs

    def generate_sql(self, user_query):
        today = datetime.now().strftime("%Y-%m-%d")

        system_prompt = f"""
            Jesteś zaawansowanym translatorem języka naturalnego na zapytania SQLite.
        
            Schemat tabeli 'documents':
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT UNIQUE NOT NULL,
                file_type TEXT,
                created_at DATETIME,
                modified_at DATETIME,
                file_size_kb REAL,
                page_count INTEGER,  -- liczba stron (pdf), slajdów (pptx/ppt) lub linii (pliki tekstowe txt/md/c/py itp.), NULL dla pozostałych
                topic TEXT,
                summary TEXT
            );
        
            Dzisiejsza data to: {today}
        
            ZASADY:
            1. Zwróć TYLKO i WYŁĄCZNIE czysty kod SQL. Zero innych słów, wstępów, ani wyjaśnień.
            2. Zapytanie musi być gotowe do wykonania w SQLite.
            3. Używaj operatora LIKE '%szukana_fraza%' (case-insensitive) do wyszukiwania tekstowego.
        
            PRZYKŁADY:
            User: Pokaż mi pliki PDF z matematyki.
            Assistant: SELECT * FROM documents WHERE file_type = 'pdf' AND topic LIKE '%matematyka%';
        
            User: Jakie mam pliki modyfikowane wczoraj?
            Assistant: SELECT * FROM documents WHERE date(modified_at) = date('{today}', '-1 day');
            
            User: W którym pliku mam historię jakiegoś miejsca?
            Assistant: SELECT * FROM documents WHERE topic LIKE '%historia%' OR summary LIKE '%historia%' OR summary LIKE '%miejsce%');
        """

        response = ollama.chat(model=self.llm_model, messages=[
            {'role': 'system', 'content': system_prompt.strip()},
            {'role': 'user', 'content': user_query}
        ])

        sql = response['message']['content'].strip()

        sql = re.sub(r"```sql\n|\n```|```", "", sql).strip()

        return sql

    def execute_query(self, sql):
        try:
            conn = self._connect()
            cursor = conn.cursor()

            cursor.execute(sql)
            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]
        except Exception as e:
            return f"BŁĄD SQL: {e}"

    def ask(self, user_query):
        print(f"\n👤 Użytkownik: {user_query}")
        print("🧠 LLM tłumaczy na SQL...")

        sql = self.generate_sql(user_query)
        print(f"💻 Wygenerowany SQL: \n   {sql}")

        print("🗄️ Wykonywanie w bazie...")
        results = self.execute_query(sql)

        if isinstance(results, str) and results.startswith("BŁĄD"):
            print(f"Brak {results}")
        elif not results:
            print("Baza poprawnie wykonała zapytanie, ale nie znalazła pasujących plików.")
        else:
            print(f"✅ Znaleziono {len(results)} pasujących dokumentów:")
            for r in results:
                print(f"    {r['filename']} | Temat: {r['topic']} | Mod: {r['modified_at']}")

        return results


if __name__ == "__main__":
    agent = TextToSQL()

    print("--- START TESTÓW TEXT-TO-SQL ---")

    agent.ask("}")
    agent.ask_semantic("Jakie mam pliki z tematu sieci neuronowych?", top_k_chunks=20, top_docs=5)