import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

client = chromadb.PersistentClient(path="./chroma_target")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

TARGET_COLLECTION_NAME = "targeting_knowledge_base"
target_collection = None

def get_or_create_target_collection():
    global target_collection
    if target_collection is not None:
        return target_collection
    try:
        target_collection = client.get_collection(
            name=TARGET_COLLECTION_NAME,
            embedding_function=embedding_function
        )
    except:
        target_collection = client.create_collection(
            name=TARGET_COLLECTION_NAME,
            embedding_function=embedding_function
        )
    return target_collection

def query_chroma_targeting(query_texts: List[str], n_results: int = 5, where_filter: Dict[str, Any] = None):
    collection = get_or_create_target_collection()
    results = collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    formatted = []
    if results["documents"]:
        for i in range(len(results["documents"][0])):
            formatted.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

    print(
        f"Queried Targeting Chroma DB for '{query_texts[0]}' "
        f"with filter {where_filter}. Found {len(formatted)} results."
    )
    return formatted

def add_document_to_target_collection(documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
    collection = get_or_create_target_collection()
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(documents)} documents to Targeting Chroma DB.")
