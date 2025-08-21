import chromadb

def get_content_from_collection(query: str, db_folder: str, collection_name: str, n_results: int = 5):
    client = chromadb.PersistentClient(path=db_folder)
    collection = client.get_collection(name=collection_name)
    results = collection.query(
                    query_texts=[query],  # Replace with your actual query
                    n_results=n_results
                )
    return results['documents'][0]
