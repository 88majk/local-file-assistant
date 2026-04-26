from pymongo import MongoClient

class MongoManager:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="magisterka_db"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["documents"]

    def clear_database(self):
        self.collection.delete_many({})
        print("Baza danych została wyczyszczona.")

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

    def migrate_attributes_to_key_value(self):
        updated_docs = 0
        cursor = self.collection.find(
            {"ai_analysis.attributes": {"$exists": True}},
            {"ai_analysis.attributes": 1}
        )

        for doc in cursor:
            current_attrs = doc.get("ai_analysis", {}).get("attributes")
            normalized = self._normalize_attributes(current_attrs)

            if current_attrs != normalized:
                self.collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"ai_analysis.attributes": normalized}}
                )
                updated_docs += 1

        if updated_docs:
            print(f"Zmigrowano atrybuty do formatu obiektu key-value w {updated_docs} dokumentach.")

        return updated_docs

    def close(self):
        self.client.close()

if __name__ == "__main__":
    db = MongoManager()
    db.clear_database()
    db.close()