# Local File Assistant

## Status projektu
Obecna architektura:
- dokumentowa baza danych MongoDB, przechowująca specyficzne informacje o pliku i jego wektorową reprezentację,
- pipeline działa w pełni lokalnie i składa się z dwóch etapów: szybka wektoryzacja + asynchroniczna analiza AI w tle,
- wyszukiwanie hybrydowe: filtry semantyczne i metadane w jednym przebiegu.

## Cel systemu
System ma umożliwiać lokalnego asystenta plików, który:
- indeksuje dokumenty z katalogu `data/` (docelowo z dowolnie podanego katalogu),
- zapisuje metadane, treść i embeddingi w MongoDB,
- wzbogaca dokument o analizę językową (topic, summary, document_type, keywords, entities),
- odpowiada na zapytania użytkownika przez hybrydowe wyszukiwanie (LLM + semantyka + filtry).

## Architektura (obecny stan)

### 1) Szybki ingest plików
Moduł: `fast_ingestion.py`

Funkcje:
- skanowanie folderu wejściowego,
- ekstrakcja tekstu przez MarkItDown,
- chunking tekstu,
- generowanie embeddingów chunków (`nomic-embed-text`),
- zapis gotowego dokumentu do kolekcji `documents` w MongoDB.

Na tym etapie dokument jest gotowy do wyszukiwania wektorowego, ale analiza semantyczna LLM jest jeszcze oznaczona jako oczekująca.

### 2) Asynchroniczna analiza AI
Moduł: `background_worker.py`

Funkcje:
- polling dokumentów o statusie `ai_analysis.status = oczekuje_na_analize`,
- pobranie próbki tekstu (`context_limit`, domyślnie 2500 znaków),
- wywołanie LLM (`Bielik-1.5B-v3.0-Instruct`) z promptem wymuszającym JSON,
- aktualizacja dokumentu o pełne pole `ai_analysis`.

Worker oddziela kosztowną analizę językową od ingestu, dzięki czemu nowe pliki pojawiają się w systemie szybciej.

### 3) Hybrydowy retriever
Moduł: `retriever.py`

Funkcje:
- parsowanie pytania użytkownika przez LLM do struktury filtrów,
- budowa zapytania MongoDB po metadanych AI,
- wektorowe dopasowanie zapytania do embeddingów chunków,
- ranking dokumentów wg podobieństwa cosinusowego,
- zwrot top wyników z cytatem i tematem.

## Model danych w MongoDB
Kolekcja: `documents`

Przykładowy dokument:

```json
{
  "filename": "wstepny-projekt.docx",
  "filepath": "data/wstepny-projekt.docx",
  "metadata": {
    "file_type": "docx",
    "size_kb": 128.42,
    "num_of_pages": 5,
    "created_at": "2026-04-13T09:12:00.000000",
    "status": "wektoryzacja_zakonczona"
  },
  "ai_analysis": {
    "status": "analiza_zakonczona",
    "topic": "...",
    "summary": "...",
    "document_type": "...",
    "keywords": ["..."],
    "entities": {
      "osoby": ["..."],
      "technologie": ["..."],
      "lokalizacje": ["..."]
    }
  },
  "chunks": [
    {
      "chunk_index": 0,
      "content": "...",
      "embedding": [0.01, -0.02, 0.03]
    }
  ]
}
```

### Indeksy
Definiowane w `db_mongo.py`:
- `ai_analysis.topic`
- `ai_analysis.keywords`

## Przepływ danych
1. `fast_ingestion.py` skanuje nowe pliki i zapisuje dokumenty z embeddingami.
2. `background_worker.py` pobiera dokumenty oczekujące i uzupełnia `ai_analysis`.
3. `retriever.py` realizuje zapytania użytkownika przez filtr + semantykę.

## Konfiguracja i zależności

### Wymagane usługi lokalne
- MongoDB dostępne pod `mongodb://localhost:27017/`
- Ollama z modelem:
  - `nomic-embed-text`

### Główne biblioteki Python
- `pymongo`
- `ollama`
- `markitdown`
- `numpy`

## Uruchomienie (aktualny workflow)
1. Wrzucenie plików do `data/`.
2. Start ingestu:

```bash
python fast_ingestion.py
```

3. Start workera AI (osobny proces):

```bash
python background_worker.py
```

4. Test zapytań:

```bash
python retriever.py
```

## Założenia projektowe na kolejny etap
Aktualna implementacja obsługuje stały zestaw pól `ai_analysis`.

Kierunek rozwoju:
- wprowadzenie mechanizmu dynamicznych atrybutów generowanych na podstawie treści,
- uczenie modelu ekstrakcji nowych par klucz-atrybut,