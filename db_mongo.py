from pymongo import MongoClient

class MongoManager:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="magisterka_db"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["documents"]

    def clear_database(self):
        self.collection.delete_many({})
        print("Baza danych została wyczyszczona.")

    def close(self):
        self.client.close()

if __name__ == "__main__":
    db = MongoManager()
    db.clear_database()
    db.close()