import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from db_mongo import MongoManager
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

class BackgroundAIWorker:
    def __init__(self, model_name="gemini-1.5-flash", context_limit=2500, max_workers=5):
        self.db = MongoManager()
        self.db.migrate_attributes_to_key_value()
        self.model_name = model_name
        self.context_limit = context_limit
        self.max_workers = max_workers

        api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        if not api_key:
            raise RuntimeError("Brak GOOGLE_API_KEY w .env")

        self.client = genai.Client(api_key=api_key)

    def get_text_sample(self, chunks):
        text_sample_list = []
        current_length = 0
        
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if not content:
                continue
                
            if current_length + len(content) > self.context_limit:
                break
                
            text_sample_list.append(content)
            current_length += len(content) + 1
        return "\n".join(text_sample_list)

    def _normalize_attributes(self, attributes):
        normalized = {}

        if isinstance(attributes, dict):
            source_items = attributes.items()
        elif isinstance(attributes, list):
            source_items = []
            for item in attributes:
                if isinstance(item, str) and ":" in item:
                    key, value = item.split(":", 1)
                    source_items.append((key, value))
                elif isinstance(item, dict):
                    if "key" in item and "value" in item:
                        source_items.append((item.get("key"), item.get("value")))
                    else:
                        source_items.extend(item.items())
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    source_items.append((item[0], item[1]))
        else:
            return normalized

        for raw_key, raw_value in source_items:
            key = str(raw_key).strip()
            value = str(raw_value).strip()
            if not key or not value:
                continue
            normalized[key] = value

        return normalized

    def _extract_json_object(self, text):
        cleaned_text = text.strip()

        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned_text)
        except Exception:
            pass

        start = cleaned_text.find("{")
        end = cleaned_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned_text[start : end + 1])

        raise ValueError("Model nie zwrócił poprawnego JSON")

    def analyze_with_llm(self, text):
        prompt = f"""
            Jesteś zaawansowanym systemem do kontekstowej ekstrakcji metadanych. Przeanalizuj tekst i zwróć wynik WYŁĄCZNIE jako obiekt JSON.

            ZASADY:
            1. Nie dodawaj żadnego tekstu, znaczników markdown ani wyjaśnień przed i po obiekcie JSON.
            2. Jeśli nie potrafisz znaleźć danych dla konkretnego pola, zostaw pustą listę [] lub wpisz "Brak".
            3. DYNAMICZNE ATRYBUTY: Pole "attributes" musi odzwierciedlać specyfikę danego dokumentu. Najpierw zidentyfikuj "document_type" (np. Faktura, Sprawozdanie, Wycena, Umowa), a następnie stwórz dedykowane parametry opisujące główny zamysł dokumentu jako obiekt key-value, np. {{"Klucz": "Wartość"}}.

            --- PRZYKŁAD 1 (Umowa) ---
            TEKST: "Umowa o dzieło zawarta 01.01.2023 w Warszawie pomiędzy Janem Kowalskim a firmą XYZ. Przedmiotem umowy jest napisanie skryptu w Pythonie. Wynagrodzenie: 5000 PLN."
            WYJŚCIE 1:
            {{
                "status": "analiza_zakonczona",
                "topic": "Umowy i Finanse",
                "summary": "Umowa o dzieło z Janem Kowalskim na stworzenie skryptu Python dla firmy XYZ.",
                "document_type": "Umowa",
                "keywords": ["umowa o dzieło", "skrypt", "wynagrodzenie"],
                "entities": {{
                    "osoby": ["Jan Kowalski"],
                    "technologie": ["Python"],
                    "lokalizacje": ["Warszawa"]
                }},
                "attributes": {{
                    "Strony_umowy": "Jan Kowalski, XYZ",
                    "Przedmiot_umowy": "napisanie skryptu",
                    "Wynagrodzenie": "5000 PLN"
                }}
            }}

            --- PRZYKŁAD 2 (Sprawozdanie studenckie) ---
            TEKST: "Sprawozdanie z laboratorium nr 3. Przedmiot: Sieci Neuronowe. Autor: Michał Nowak. Zadanie polegało na implementacji sieci konwolucyjnej do rozpoznawania obrazów z użyciem biblioteki TensorFlow."
            WYJŚCIE 2:
            {{
                "status": "analiza_zakonczona",
                "topic": "Uczenie Maszynowe",
                "summary": "Raport Michała Nowaka z implementacji sieci konwolucyjnej w TensorFlow na zajęcia z Sieci Neuronowych.",
                "document_type": "Sprawozdanie akademickie",
                "keywords": ["laboratorium", "rozpoznawanie obrazów", "CNN"],
                "entities": {{
                    "osoby": ["Michał Nowak"],
                    "technologie": ["TensorFlow"],
                    "lokalizacje": []
                }},
                "attributes": {{
                    "Przedmiot_akademicki": "Sieci Neuronowe",
                    "Temat_ćwiczenia": "implementacja sieci konwolucyjnej",
                    "Rodzaj_zadania": "rozpoznawanie obrazów"
                }}
            }}

            --- PRZYKŁAD 3 (Dokument księgowy/Faktura) ---
            TEKST: "Faktura VAT nr 12/2023. Sprzedawca: Hurtownia Budowlana BUD-MAX. Nabywca: Jan Budowniczy. Towar: 100x Pustak ceramiczny. Kwota do zapłaty: 1500,00 zł."
            WYJŚCIE 3:
            {{
                "status": "analiza_zakonczona",
                "topic": "Zakupy budowlane",
                "summary": "Faktura za zakup pustaków ceramicznych od firmy BUD-MAX przez Jana Budowniczego.",
                "document_type": "Faktura VAT",
                "keywords": ["materiały budowlane", "rozliczenie"],
                "entities": {{
                    "osoby": ["Jan Budowniczy"],
                    "technologie": [],
                    "lokalizacje": []
                }},
                "attributes": {{
                    "Sprzedawca": "BUD-MAX",
                    "Rodzaj_towaru": "Pustak ceramiczny",
                    "Kwota_transakcji": "1500,00 zł"
                }}
            }}

            --- TWÓJ CEL ---
            TEKST DO ANALIZY:
            {text}
        """

        try:
            response = self.client.models.generate_content(
                model='gemma-4-31b-it', 
                contents=prompt + text,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            result = self._extract_json_object(response.text or "")
            result["status"] = "analiza_zakonczona"
            result["attributes"] = self._normalize_attributes(result.get("attributes", {}))
            return result
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "503" in error_msg:
                print(f"Serwer API zajęty. Czekam 30 sekund...")
                time.sleep(30)
                return self.analyze_with_llm(text)
            
            print(f"Błąd LLM: {error_msg}")
            return {"status": "błąd_analizy", "attributes": {}}

    def _process_single_doc(self, doc):
        start_time = time.time()
        
        text_sample = self.get_text_sample(doc.get("chunks", []))
        ai_result = self.analyze_with_llm(text_sample)

        if ai_result.get("status") == "analiza_zakonczona":
            self.db.collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"ai_analysis": ai_result}}
            )
            elapsed = time.time() - start_time
            return f"[{elapsed:.1f}s] Zakończono: {doc['filename']} | Temat: {ai_result.get('topic')}"
        else:
            self.db.collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"ai_analysis.status": "błąd_analizy"}}
            )
            return f"Analiza zakończona błędem dla: {doc['filename']}"

    def run_worker(self):
        print(f"Uruchamiam RÓWNOLEGŁEGO workera AI (Model: {self.model_name}, Wątki: {self.max_workers}). Oczekiwanie na dokumenty...")
        
        while True:
            docs_to_process = []
            
            for _ in range(self.max_workers):
                doc = self.db.collection.find_one_and_update(
                    {"ai_analysis.status": "oczekuje_na_analize"},
                    {"$set": {"ai_analysis.status": "w_trakcie_analizy"}}
                )
                if doc:
                    docs_to_process.append(doc)

            if not docs_to_process:
                time.sleep(10)
                continue

            print(f"\nPobrano {len(docs_to_process)} plików do jednoczesnej analizy...")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._process_single_doc, doc) for doc in docs_to_process]
                
                for future in as_completed(futures):
                    print(future.result())


if __name__ == "__main__":
    worker = BackgroundAIWorker(model_name="gemma-4-31b-it", max_workers=5)
    try:
        worker.run_worker()
    except KeyboardInterrupt:
        print("\nZatrzymano workera AI.")