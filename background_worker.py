import time
import json
from db_mongo import MongoManager
from local_llm import LocalBielikLLM

class BackgroundAIWorker:
    def __init__(self, model_name="speakleash/Bielik-1.5B-v3.0-Instruct", context_limit=2500):
        self.db = MongoManager()
        self.model_name = model_name
        self.context_limit = context_limit
        self.llm = LocalBielikLLM(model_id=self.model_name)

    def get_text_sample(self, chunks):
        """Pobiera tylko początek dokumentu (do limitu znaków), by oszczędzić czas LLM."""
        text_sample = ""
        for chunk in chunks:
            if len(text_sample) >= self.context_limit:
                break
            text_sample += chunk.get("content", "") + "\n"
        return text_sample[:self.context_limit]

    def analyze_with_llm(self, text):
        """Wysyła zapytanie do LLM z użyciem techniki Few-Shot Prompting."""
        prompt = f"""
            Jesteś precyzyjnym asystentem ekstrakcji danych. Przeanalizuj poniższy tekst i zwróć wynik WYŁĄCZNIE jako obiekt JSON.

            ZASADY:
            1. Nie dodawaj żadnego tekstu przed ani po JSON-ie.
            2. Jeśli nie potrafisz znaleźć danych, zostaw puste listy lub wpisz "Brak".

            PRZYKŁAD WEJŚCIA:
            "Umowa o dzieło zawarta dnia 01.01.2023 w Warszawie pomiędzy Janem Kowalskim a firmą XYZ. Przedmiotem umowy jest napisanie skryptu w Pythonie. Wynagrodzenie: 5000 PLN."

            PRZYKŁAD WYJŚCIA:
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
                }}
            }}

            TEKST DO ANALIZY:
            {text}
        """

        try:
            result = self.llm.generate_json(prompt)
            result["status"] = "analiza_zakonczona" 
            return result
        except Exception as e:
            print(f"Błąd LLM: {e}")
            return {"status": "błąd_analizy"}

    def run_worker(self):
        print(f"Uruchamiam workera AI (Model: {self.model_name}). Oczekiwanie na dokumenty...")
        
        while True:
            doc = self.db.collection.find_one({"ai_analysis.status": "oczekuje_na_analize"})
            
            if not doc:
                time.sleep(10) # sleep worker, to avoid cpu usage
                continue

            print(f"\nRozpoczynam głęboką analizę pliku: {doc['filename']}")
            start_time = time.time()

            # 2. Przygotuj skrócony tekst
            text_sample = self.get_text_sample(doc.get("chunks", []))

            # 3. Wywołaj ciężki model AI
            ai_result = self.analyze_with_llm(text_sample)

            if ai_result.get("status") == "analiza_zakonczona":
                # 4. Zaktualizuj dokument w MongoDB o nowe, bogate metadane
                self.db.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"ai_analysis": ai_result}}
                )
                elapsed = time.time() - start_time
                print(f"Zakończono sukcesem w {elapsed:.1f}s. Temat: {ai_result.get('topic')}")
            else:
                # Oznacz dokument jako błędny, aby worker nie zapętlił się na nim w nieskończoność
                self.db.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"ai_analysis.status": "błąd_analizy"}}
                )
                print("Analiza zakończona błędem.")

if __name__ == "__main__":
    worker = BackgroundAIWorker()
    try:
        worker.run_worker()
    except KeyboardInterrupt:
        print("\nZatrzymano workera AI.")