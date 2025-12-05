import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_utils_target import add_document_to_target_collection, get_or_create_target_collection



PDF_FOLDER = "./target_pdfs"

# ------------------------------------------------------
# 1. PDF 로드 (신규 파일만)
# ------------------------------------------------------
def load_pdfs():
    docs = []

    print("PDF 폴더 스캔 중...")
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise Exception("target_pdfs 폴더에 PDF 파일이 없습니다.")

    print(f"폴더 내 모든 PDF: {pdf_files}")

    # --- 기존 DB에서 이미 저장된 filename 목록 가져오기 ---
    collection = get_or_create_target_collection()
    existing_files = set()

    if collection is not None:
        data = collection.get(include=["metadatas"])
        for item in data["metadatas"]:
            if item and "filename" in item:
                existing_files.add(item["filename"])

    print("이미 인덱싱된 파일:", existing_files)

    # --- 신규 PDF만 선택 ---
    new_pdfs = [f for f in pdf_files if f not in existing_files]

    if not new_pdfs:
        print("➡️ 새로운 PDF 없음. 인덱싱 스킵!")
        return docs

    print("신규 인덱싱 대상 PDF:", new_pdfs)

    # --- 신규 PDF 로드 ---
    for pdf in new_pdfs:
        pdf_path = os.path.join(PDF_FOLDER, pdf)
        print(f"\n📄 로드 중: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()

        print(f"- 페이지 수: {len(pdf_docs)}")

        # meta 정보에 filename 저장
        for d in pdf_docs:
            d.metadata["filename"] = pdf

        docs.extend(pdf_docs)

    print(f"\n신규 로드된 문서 총 수: {len(docs)}개")
    return docs


# ------------------------------------------------------
# 2. 청크 생성
# ------------------------------------------------------
def split_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)
    print(f"생성된 전체 청크: {len(chunks)}")
    return chunks


# ------------------------------------------------------
# 3. ChromaDB 저장
# ------------------------------------------------------
def save_to_chroma(chunks):
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [
        {
            "source_type": "타겟분류",
            "filename": chunk.metadata.get("filename", "unknown"),
            "page": chunk.metadata.get("page", None)
        }
        for chunk in chunks
    ]
    ids = [f"target_chunk_{i}" for i in range(len(chunks))]

    print("ChromaDB 저장 중...")
    add_document_to_target_collection(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print("저장 완료!")


# ------------------------------------------------------
# 4. 실행
# ------------------------------------------------------
if __name__ == "__main__":
    print("PDF 로드 중...")
    docs = load_pdfs()

    if not docs:
        print("신규 문서 없음 → 프로그램 종료.")
        exit()

    print("청크 분할 중...")
    chunks = split_chunks(docs)

    print("DB 저장 시작")
    save_to_chroma(chunks)

    print("\n전체 ingest 완료!")
