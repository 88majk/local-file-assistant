import json
import re
import os
import time
from datetime import datetime
import numpy as np
import ollama
import spacy
nlp = spacy.load("pl_core_news_sm")
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from db_mongo import MongoManager

class Retriever:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="magisterka_db"):
        self.db = MongoManager(uri=mongo_uri, db_name=db_name)
        self.db.migrate_attributes_to_key_value()
        self.collection = self.db.collection

        api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        if not api_key:
            raise RuntimeError("Brak GOOGLE_AI_STUDIO_API_KEY w .env")
            
        self.genai_client = genai.Client(api_key=api_key)
        self.llm_model = "gemma-4-31b-it"
        self.embed_model = "nomic-embed-text" 

    def close(self):
        self.db.close()

    def lemmatize_text(self, text):
        doc = nlp(text)
        return doc[0].lemma_ if doc else text

    def _get_database_context(self):
        try:
            doc_types = self.collection.distinct("ai_analysis.document_type")
            doc_types = [dt for dt in doc_types if dt]

            sample_docs = self.collection.find(
                {"ai_analysis.status": "analiza_zakonczona"}, 
                {
                    "chunks": 0,
                    "_id": 0
                }
            ).limit(50)
            
            attr_keys = set()
            for doc in sample_docs:
                attrs = doc.get("ai_analysis", {}).get("attributes", {})
                if isinstance(attrs, dict):
                    for key in attrs.keys():
                        if key:
                            attr_keys.add(str(key).strip())
                elif isinstance(attrs, list):
                    for attr in attrs:
                        if isinstance(attr, str) and ":" in attr:
                            attr_keys.add(attr.split(":", 1)[0].strip())
                        elif isinstance(attr, dict):
                            key = str(attr.get("key", "")).strip()
                            if key:
                                attr_keys.add(key)

            return list(doc_types), list(attr_keys)
        except Exception as e:
            print(f"[Ostrzeżenie] Nie udało się pobrać kontekstu bazy: {e}")
            return [], []

    def _normalize_iso_datetime(self, raw_value, end_of_day=False):
        if not raw_value:
            return None

        value = str(raw_value).strip()
        if not value:
            return None

        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            try:
                parsed = datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                return None

        if end_of_day and parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0 and parsed.microsecond == 0:
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)

        return parsed.isoformat()

    def _safe_int(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_user_query_with_llm(self, user_query):
        available_types, available_attr_keys = self._get_database_context()
        
        types_str = ", ".join(available_types) if available_types else "Brak zdefiniowanych (zgaduj)"
        attrs_str = ", ".join(available_attr_keys) if available_attr_keys else "Brak zdefiniowanych"

        prompt = f"""
            Jesteś analitycznym systemem mapującym zapytania na bazę NoSQL.
            Wyciągnij intencję użytkownika, filtry twarde i zapytanie wektorowe.

            PRZYKŁADOWA STRUKTURA POJEDYNCZEGO DOKUMENTU:
            {{
                "filename": "faktura_xkom.pdf",
                "metadata": {{
                    "created_at": "2026-04-13T09:12:00",
                    "num_of_pages": 3
                }},
                "ai_analysis": {{
                    "document_type": "Faktura VAT",
                    "attributes": {{
                        "Sprzedawca": "X-KOM",
                        "Kwota": "5000 PLN"
                    }}
                }}
            }}

            ZASADY:
            1. Zwróć WYŁĄCZNIE obiekt JSON.
            2. Przy dopasowywaniu "filter_type", postaraj się użyć JEDNEJ Z NASTĘPUJĄCYCH dostępnych w bazie wartości: [{types_str}]. Jeśli żadna nie pasuje, zwróć null.
            3. "filter_keywords" to miejsce na OGÓLNE kategorie, tematy i niesprecyzowane frazy (np. "elektronika", "programowanie", "zakupy").
            4. "attribute_filters" ma być obiektem mapującym pary klucz:wartość, np. {{"Sprzedawca": "X-KOM", "Kwota": "5000"}}.
               Rezerwuj je TYLKO na precyzyjne parametry i używaj kluczy z listy: [{attrs_str}].
            5. ZŁOTA ZASADA: Jeśli użytkownik zadaje ogólne pytanie i nie szuka precyzyjnej wartości atrybutu, zostaw "attribute_filters" jako pusty obiekt {{}}. Lepiej zostawić puste niż zgadywać.
            6. Technologie dotyczą tylko specyficznych technologii, takich jak narzędzia/programy. Nie umieszczaj tam ogólnych słów kluczowych.
            7. Jeśli zapytanie dotyczy daty dodania dokumentu, uzupełnij pola:
               - "created_at_from": data ISO (np. "2026-03-01")
               - "created_at_to": data ISO (np. "2026-03-31")
               Gdy brak filtra daty, zwróć null dla obu pól.
            8. Jeśli zapytanie dotyczy długości pliku, zwróć:
               - "file_length_filter": {{"metric": "num_of_pages" | "num_of_slides" | "any", "min": liczba|null, "max": liczba|null}}
               Gdy brak filtra długości, zwróć null.

            PRZYKŁAD 1 (Zapytanie ogólne - brak atrybutów):
            Dostępne typy: [Umowa, Sprawozdanie, Faktura VAT]
            Pytanie: "Szukam umowy o dzieło z Janem Kowalskim dotyczącej Pythona"
            JSON: {{
                "semantic_query": "umowa o dzieło warunki wynagrodzenie",
                "filter_type": "Umowa",
                "filter_person": "Jan Kowalski",
                "filter_tech": "Python",
                "filter_keywords": ["umowa o dzieło", "python"],
                "attribute_filters": {{}},
                "created_at_from": null,
                "created_at_to": null,
                "file_length_filter": null
            }}

            PRZYKŁAD 2 (Zapytanie ogólne tematyczne - brak atrybutów):
            Dostępne typy: [Umowa, Sprawozdanie, Faktura VAT]
            Pytanie: "Jakie posiadam faktury za zakup elektroniki?"
            JSON: {{
                "semantic_query": "faktury za zakup elektroniki",
                "filter_type": "Faktura VAT",
                "filter_person": null,
                "filter_tech": null,
                "filter_keywords": [ "elektronika"],
                "attribute_filters": {{}},
                "created_at_from": null,
                "created_at_to": null,
                "file_length_filter": null
            }}

            PRZYKŁAD 3 (Zapytanie precyzyjne - użycie atrybutów):
            Dostępne typy: [Umowa, Sprawozdanie, Faktura VAT]
            Dostępne klucze: [Sprzedawca, Miasto, Kwota]
            Pytanie: "Pokaż faktury od firmy X-KOM na kwotę 5000"
            JSON: {{
                "semantic_query": "faktury od firmy x-kom na kwotę 5000",
                "filter_type": "Faktura VAT",
                "filter_person": null,
                "filter_tech": null,
                "filter_keywords": ["X-KOM"],
                "attribute_filters": {{
                    "Sprzedawca": "X-KOM",
                    "Kwota": "5000"
                }},
                "created_at_from": null,
                "created_at_to": null,
                "file_length_filter": null
            }}

            PRZYKŁAD 4 (Data + długość):
            Pytanie: "Znajdź prezentacje dodane po 2026-03-01 mające co najmniej 20 slajdów"
            JSON: {{
                "semantic_query": "prezentacje dodane po marcu 2026",
                "filter_type": null,
                "filter_person": null,
                "filter_tech": null,
                "filter_keywords": ["prezentacje"],
                "attribute_filters": {{}},
                "created_at_from": "2026-03-01",
                "created_at_to": null,
                "file_length_filter": {{
                    "metric": "num_of_slides",
                    "min": 20,
                    "max": null
                }}
            }}

            --- TWÓJ CEL ---
            Dostępne typy w bazie: [{types_str}]
            Dostępne klucze atrybutów: [{attrs_str}]
            
            Pytanie użytkownika: "{user_query}"
        """
        try:
            response = self.genai_client.models.generate_content(
                model=self.llm_model,
                contents=prompt.strip(),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
                
            return json.loads(cleaned_text)
            
        except Exception as e:
            print(f"[Ostrzeżenie] Nie udało się przeparsować zapytania chmurowo: {e}")
            return {
                "semantic_query": user_query,
                "filter_type": None,
                "filter_person": None,
                "filter_tech": None,
                "filter_keywords": None,
                "attribute_filters": {},
                "created_at_from": None,
                "created_at_to": None,
                "file_length_filter": None,
            }

    def _build_mongo_filter(self, search_params):
        mongo_query = {}

        mongo_query["ai_analysis.status"] = "analiza_zakonczona"
        attribute_filters = search_params.get("attribute_filters", {})

        if search_params.get("filter_type"):
            mongo_query["ai_analysis.document_type"] = {
                "$regex": search_params["filter_type"],
                "$options": "i",
            }

        if search_params.get("filter_person"):
            mongo_query["ai_analysis.entities.osoby"] = {
                "$regex": search_params["filter_person"],
                "$options": "i",
            }

        if search_params.get("filter_tech"):
            mongo_query["ai_analysis.entities.technologie"] = {
                "$regex": search_params["filter_tech"],
                "$options": "i",
            }
        
        if search_params.get("filter_keywords"):
            valid_keywords = [k.strip() for k in search_params["filter_keywords"] if isinstance(k, str) and k.strip()]
            
            if valid_keywords:
                regex_list = [re.compile(re.escape(self.lemmatize_text(k)), re.IGNORECASE) for k in valid_keywords]
                
                mongo_query["ai_analysis.keywords"] = {
                    "$in": regex_list
                }

        created_at_from = self._normalize_iso_datetime(search_params.get("created_at_from"), end_of_day=False)
        created_at_to = self._normalize_iso_datetime(search_params.get("created_at_to"), end_of_day=True)
        if created_at_from or created_at_to:
            date_query = {}
            if created_at_from:
                date_query["$gte"] = created_at_from
            if created_at_to:
                date_query["$lte"] = created_at_to
            mongo_query["metadata.created_at"] = date_query

        file_length_filter = search_params.get("file_length_filter")
        if isinstance(file_length_filter, dict):
            metric = str(file_length_filter.get("metric", "any")).strip().lower()
            min_length = self._safe_int(file_length_filter.get("min"))
            max_length = self._safe_int(file_length_filter.get("max"))

            if min_length is not None or max_length is not None:
                range_query = {}
                if min_length is not None:
                    range_query["$gte"] = min_length
                if max_length is not None:
                    range_query["$lte"] = max_length

                if metric in {"num_of_pages", "num_of_slides"}:
                    mongo_query[f"metadata.{metric}"] = range_query
                else:
                    mongo_query["$or"] = [
                        {"metadata.num_of_pages": range_query},
                        {"metadata.num_of_slides": range_query},
                    ]

        if attribute_filters:
            if "$and" not in mongo_query:
                mongo_query["$and"] = []

            if isinstance(attribute_filters, dict):
                for raw_key, raw_value in attribute_filters.items():
                    key = str(raw_key).strip()
                    value = str(raw_value).strip()
                    if not key or not value:
                        continue

                    mongo_query["$and"].append({
                        f"ai_analysis.attributes.{key}": {
                            "$regex": re.escape(value),
                            "$options": "i",
                        }
                    })
            elif isinstance(attribute_filters, list):
                for item in attribute_filters:
                    if isinstance(item, str) and ":" in item:
                        key, value = item.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        if key and value:
                            mongo_query["$and"].append({
                                f"ai_analysis.attributes.{key}": {
                                    "$regex": re.escape(value),
                                    "$options": "i",
                                }
                            })

        return mongo_query

    def cosine_similarity(self, v1, v2):
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def search(self, user_query, top_k_docs=10, use_embeddings=True):
        print(f"\n[{user_query}] -> Rozpoczynam wyszukiwanie...")

        search_params = self._parse_user_query_with_llm(user_query)
        print(f"\nZrozumiana intencja: {json.dumps(search_params, indent=2, ensure_ascii=False)}")

        mongo_filter = self._build_mongo_filter(search_params)
        print(f"\nZbudowany filtr MongoDB: {json.dumps(mongo_filter, indent=2, ensure_ascii=False, default=str)}")
        candidate_docs = list(self.collection.find(mongo_filter))

        if not candidate_docs:
            print("❌ Brak dopasowań w bazie na podstawie metadanych.")
            return []

        if not use_embeddings:
            results = []
            for doc in candidate_docs[:top_k_docs]:
                chunk_text = doc.get("chunks", [{"content": ""}])[0].get("content", "")
                results.append({
                    "filename": doc["filename"],
                    "filepath": doc.get("filepath", ""),
                    "topic": doc["ai_analysis"].get("topic", "Brak tematu"),
                    "document_type": doc["ai_analysis"].get("document_type", "Brak typu"),
                    "score": 1.0,
                    "snippet": chunk_text[:250].replace("\n", " ") + "...",
                })
                
            print("\nNAJLEPSZE DOPASOWANIA:")
            for i, res in enumerate(results, 1):
                file_url = "file:///" + res.get('filepath', '').replace('\\', '/').lstrip('/')
                clickable_name = f"\033]8;;{file_url}\033\\{res['filename']}\033]8;;\033\\"
                
                print(f"{i}. {clickable_name}")
                print(f"   Temat: {res['topic']}")
                print(f"   Typ dokumentu: {res['document_type']}")
            return results

        semantic_query = search_params.get("semantic_query") or user_query
        try:
            query_embedding = ollama.embed(
                model=self.embed_model, input=[semantic_query]
            )["embeddings"][0]
        except Exception as e:
            print(f"❌ Błąd generowania wektora zapytania: {e}")
            return []

        results = []
        for doc in candidate_docs:
            best_chunk_score = -1

            for chunk in doc.get("chunks", []):
                chunk_emb = chunk.get("embedding")
                if not chunk_emb:
                    continue

                score = self.cosine_similarity(query_embedding, chunk_emb)

                if score > best_chunk_score:
                    best_chunk_score = score

            if best_chunk_score > 0.4:
                results.append(
                    {
                        "filename": doc["filename"],
                        "filepath": doc.get("filepath", ""),
                        "topic": doc["ai_analysis"].get("topic", "Brak tematu"),
                        "score": best_chunk_score,
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k_docs]

        print("\nNAJLEPSZE DOPASOWANIA:")
        for i, res in enumerate(top_results, 1):
            # Kodowanie linku dla terminala (OSC 8) - umożliwia Ctrl+Klik w VS Code
            file_url = "file:///" + res.get('filepath', '').replace('\\', '/').lstrip('/')
            clickable_name = f"\033]8;;{file_url}\033\\{res['filename']}\033]8;;\033\\"
            
            print(f"{i}. {clickable_name} [Trafność semantyczna: {res['score']:.3f}]")
            print(f"   Temat: {res['topic']}")

        return top_results

if __name__ == "__main__":
    retriever = Retriever()

    zapytania_testowe = [
        # "Jakie posiadam faktury w folderze?",
        # "Jakie posiadam faktury za zakup elektroniki?",
        # "Szukam faktury za perfumy Yves Saint Laurent",
        # "Szukam pliku z wyceną strony internetowej",
        "Szukam pliku z odpowiedzami do przedmiotu unijne prawo gospodarcze",
        "Prezentacje, mające mniej niż 10 slajdów, dodane w 2025 roku",
    ]

    for pytanie in zapytania_testowe:
        retriever.search(pytanie, use_embeddings=False)

    retriever.close()