# c:\Users\choij\Desktop\일경험사업\dev_code\ai\rag_utils.py

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any

# Chroma DB 클라이언트 초기화 (영속성 모드)
# 데이터를 디스크에 저장하여 서버가 재시작되어도 데이터가 유지됩니다.
# 'chroma_data' 폴더에 데이터가 저장됩니다.
client = chromadb.PersistentClient(path="./chroma_data")

# 임베딩 함수 설정
# 'all-MiniLM-L6-v2' 모델은 영어에 최적화되어 있지만,
# 한국어 모델(예: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')로 교체 가능합니다.
# 한국어 모델 사용 시, 'sentence-transformers' 라이브러리가 해당 모델을 다운로드합니다.
# 여기서는 예시로 'all-MiniLM-L6-v2'를 사용하고, 실제 한국어 서비스에서는 적절한 한국어 모델로 변경해야 합니다.
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 컬렉션 이름 정의
COLLECTION_NAME = "marketing_knowledge_base"

# collection 객체를 모듈 레벨에서 초기화하여 다른 파일에서 직접 임포트할 수 있도록 합니다.
collection = None # 초기화
def get_or_create_collection():
    """
    Chroma DB 컬렉션을 가져오거나 새로 생성합니다.
    """
    global collection # 전역 collection 변수를 사용함을 명시
    if collection is not None:
        return collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
        print(f"Chroma DB collection '{COLLECTION_NAME}' loaded.")
    except:
        collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
        print(f"Chroma DB collection '{COLLECTION_NAME}' created.")
    return collection

# 모듈 로드 시점에 collection을 초기화합니다.
collection = get_or_create_collection()

def add_document_to_chroma(documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
    """
    문서를 Chroma DB 컬렉션에 추가합니다.
    :param documents: 추가할 문서 텍스트 리스트
    :param metadatas: 각 문서에 대한 메타데이터 리스트 (딕셔너리 형태)
    :param ids: 각 문서의 고유 ID 리스트
    """
    # collection = get_or_create_collection() # 이미 모듈 레벨에서 초기화되었으므로 필요 없음
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Added {len(documents)} documents to Chroma DB.")

def query_chroma(query_texts: List[str], n_results: int = 5, where_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Chroma DB 컬렉션에서 쿼리 텍스트와 유사한 문서를 검색합니다.
    :param query_texts: 검색할 쿼리 텍스트 리스트
    :param n_results: 반환할 결과의 최대 개수
    :param where_filter: 메타데이터 필터링 조건 (예: {"source_type": "정책"})
    :return: 검색된 문서, 메타데이터, 거리 정보를 포함하는 딕셔너리 리스트
    """
    # collection = get_or_create_collection() # 이미 모듈 레벨에서 초기화되었으므로 필요 없음
    results = collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where_filter,
        include=['documents', 'metadatas', 'distances']
    )
    
    # 결과를 좀 더 사용하기 쉬운 형태로 가공
    formatted_results = []
    if results['documents']:
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
    
    print(f"Queried Chroma DB for '{query_texts[0]}' with filter {where_filter}. Found {len(formatted_results)} results.")
    return formatted_results

def get_all_documents_from_chroma(where_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Chroma DB 컬렉션에서 모든 문서를 가져옵니다.
    :param where_filter: 메타데이터 필터링 조건 (선택 사항)
    """
    # collection = get_or_create_collection() # 이미 모듈 레벨에서 초기화되었으므로 필요 없음
    # get() 메서드를 사용하여 모든 항목을 가져옵니다.
    # where 필터를 추가하여 필터링된 결과를 가져올 수 있도록 합니다.
    results = collection.get(where=where_filter, include=['documents', 'metadatas'])
    
    # 결과를 가공합니다.
    formatted_results = []
    if results['ids']:
        for i in range(len(results['ids'])):
            formatted_results.append({
                "id": results['ids'][i],
                "document": results['documents'][i],
                "metadata": results['metadatas'][i]
            })
    
    print(f"Retrieved {len(formatted_results)} documents from Chroma DB with filter {where_filter}.")
    return formatted_results

def get_document_by_id(doc_id: str) -> Dict[str, Any] | None:
    """
    ID를 사용하여 Chroma DB에서 단일 문서를 가져옵니다.
    """
    # collection = get_or_create_collection() # 이미 모듈 레벨에서 초기화되었으므로 필요 없음
    result = collection.get(ids=[doc_id], include=['documents', 'metadatas'])
    
    if result and result['ids']:
        return {
            "id": result['ids'][0],
            "document": result['documents'][0],
            "metadata": result['metadatas'][0]
        }
    print(f"Document with id '{doc_id}' not found.")
    return None

def update_document_in_chroma(doc_id: str, document: str, metadata: Dict[str, Any]):
    """
    Chroma DB의 문서를 업데이트(upsert)합니다.
    """
    # collection = get_or_create_collection() # 이미 모듈 레벨에서 초기화되었으므로 필요 없음
    collection.upsert(
        ids=[doc_id],
        documents=[document],
        metadatas=[metadata]
    )
    print(f"Upserted document with id '{doc_id}'.")

def delete_document_from_chroma(doc_id: str):
    """
    ID를 사용하여 Chroma DB에서 문서를 삭제합니다.
    """
    # collection = get_or_create_collection() # 이미 모듈 레벨에서 초기화되었으므로 필요 없음
    collection.delete(ids=[doc_id])
    print(f"Deleted document with id '{doc_id}'.")

# 예시: Chroma DB에 문서 추가 및 쿼리
if __name__ == "__main__":
    # 컬렉션 초기화 (기존 데이터 삭제 후 새로 시작하려면 주석 해제)
    # client.delete_collection(name=COLLECTION_NAME)
    # collection = get_or_create_collection()

    # 문서 추가 예시
    add_document_to_chroma(
        documents=[
            "KT 5G 프리미엄 요금제는 데이터 완전 무제한 혜택을 제공합니다.",
            "가족 결합 시 통신 요금 할인이 적용됩니다.",
            "20대 대학생 타겟 마케팅 성공 사례: 최신 스마트폰 프로모션과 데이터 무제한 결합.",
            "스팸 메시지 발송 규정: 특수문자 과다 사용 금지, 긴급성 강조 문구 지양.",
            "40대 주부 타겟 마케팅 성공 사례: 키즈 콘텐츠 할인 및 가족 통신비 절감 혜택 강조."
        ],
        metadatas=[
            {"source_type": "정책", "title": "5G 프리미엄 요금제"},
            {"source_type": "정책", "title": "가족 결합 할인"},
            {"source_type": "성공 사례", "target_group": "20대 대학생"},
            {"source_type": "스팸/광고 정책", "rule_id": "SPAM-001"},
            {"source_type": "성공 사례", "target_group": "40대 주부"}
        ],
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )

    # 문서 쿼리 예시
    print("\n--- 정책 검색 ---")
    policy_results = query_chroma(query_texts=["5G 요금제 혜택은 무엇인가요?"], where_filter={"source_type": "정책"})
    for res in policy_results:
        print(f"Document: {res['document']}, Metadata: {res['metadata']}, Distance: {res['distance']:.2f}")

    print("\n--- 성공 사례 검색 (20대 대학생) ---")
    success_case_results = query_chroma(query_texts=["대학생에게 효과적인 마케팅 메시지는?"], where_filter={"$and": [{"source_type": "성공 사례"}, {"target_group": "20대 대학생"}]})
    for res in success_case_results:
        print(f"Document: {res['document']}, Metadata: {res['metadata']}, Distance: {res['distance']:.2f}")

    print("\n--- 모든 문서 조회 ---")
    all_docs = get_all_documents_from_chroma()
    if all_docs:
        for doc in all_docs:
            print(f"ID: {doc['id']}, Document: {doc['document']}, Metadata: {doc['metadata']}")
    else:
        print("Chroma DB에 문서가 없습니다.")
