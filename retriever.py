import json
import numpy as np
import ollama
from pymongo import MongoClient
from local_llm import LocalBielikLLM


class StandaloneHybridRetriever:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="magisterka_db"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["documents"]

        self.llm_model = "speakleash/Bielik-1.5B-v3.0-Instruct"
        self.llm = LocalBielikLLM(model_id=self.llm_model)
        self.embed_model = (
            "nomic-embed-text"  
        )

    def close(self):
        """Zamyka połączenie z bazą."""
        self.client.close()

    def _parse_user_query_with_llm(self, user_query):
        """Używa LLM do przekształcenia naturalnego pytania na filtry bazy danych."""
        prompt = f"""
            Jesteś systemem tłumaczącym zapytania użytkownika na parametry wyszukiwania bazy danych.
            Twoim zadaniem jest wyciągnąć z zapytania potencjalne filtry i stworzyć zoptymalizowane hasło do wyszukiwania wektorowego.

            Zwróć WYŁĄCZNIE obiekt JSON. Jeśli nie potrafisz wyciągnąć danego filtru, zwróć null.

            Przykład 1:
            Pytanie: "Szukam umowy o dzieło z Janem Kowalskim dotyczącej Pythona"
            JSON: {{
                "semantic_query": "umowa o dzieło warunki wynagrodzenie",
                "filter_type": "Umowa",
                "filter_person": "Jan Kowalski",
                "filter_tech": "Python"
            }}

            Przykład 2:
            Pytanie: "Jakie są podstawy sieci neuronowych?"
            JSON: {{
                "semantic_query": "podstawy sieci neuronowych uczenie maszynowe",
                "filter_type": null,
                "filter_person": null,
                "filter_tech": null
            }}

            Pytanie użytkownika: "{user_query}"
        """
        try:
            return self.llm.generate_json(prompt)
        except Exception as e:
            print(f"[Ostrzeżenie] Nie udało się przeparsować zapytania: {e}")
            return {
                "semantic_query": user_query,
                "filter_type": None,
                "filter_person": None,
                "filter_tech": None,
            }

    def _build_mongo_filter(self, search_params):
        """Buduje natywne zapytanie MongoDB na podstawie wyciągniętych przez LLM filtrów."""
        mongo_query = {}

        mongo_query["ai_analysis.status"] = "analiza_zakonczona"

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

        return mongo_query

    def cosine_similarity(self, v1, v2):
        """Matematyczne obliczenie podobieństwa między dwoma wektorami."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def search(self, user_query, top_k_docs=3):
        print(f"\n[{user_query}] -> Rozpoczynam wyszukiwanie hybrydowe...")

        search_params = self._parse_user_query_with_llm(user_query)

        mongo_filter = self._build_mongo_filter(search_params)
        candidate_docs = list(self.collection.find(mongo_filter))

        if not candidate_docs:
            print(
                "❌ MongoDB odrzuciło wszystkie dokumenty (brak dopasowań w metadanych)."
            )
            return []

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
            best_chunk_text = ""

            for chunk in doc.get("chunks", []):
                chunk_emb = chunk.get("embedding")
                if not chunk_emb:
                    continue

                score = self.cosine_similarity(query_embedding, chunk_emb)

                if score > best_chunk_score:
                    best_chunk_score = score
                    best_chunk_text = chunk["content"]

            if best_chunk_score > -1:
                results.append(
                    {
                        "filename": doc["filename"],
                        "topic": doc["ai_analysis"].get("topic", "Brak tematu"),
                        "score": best_chunk_score,
                        "snippet": best_chunk_text[:250].replace("\n", " ") + "...",
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k_docs]

        print("\n🏆 NAJLEPSZE DOPASOWANIA:")
        for i, res in enumerate(top_results, 1):
            print(f"{i}. {res['filename']} [Trafność semantyczna: {res['score']:.3f}]")
            print(f"   Temat: {res['topic']}")
            print(f"   Cytat: {res['snippet']}\n")

        return top_results


if __name__ == "__main__":
    retriever = StandaloneHybridRetriever()

    zapytania_testowe = [
        "Jakie są główne założenia sztucznej inteligencji?",
        "Znajdź mi faktury od firmy XYZ",
        "W jakim pliku mam skrypt do sortowania bąbelkowego?",
    ]

    for pytanie in zapytania_testowe:
        retriever.search(pytanie)

    retriever.close()
