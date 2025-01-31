import json
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 임베딩 모델 설정 (Sentence-BERT 계열)
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Text Splitter 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 청크 크기 (문자 수 기준)
    chunk_overlap=50  # 중첩 길이
)

def load_and_prepare_documents(json_file):
    """
    간단한 예시로,
    JSON 구조:
      {
        "채용공고목록": [
          {
            "공고번호": 1,
            "채용제목": "...",
            "회사명": "...",
            ...
          },
          ...
        ]
      }
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    docs = []
    for item in data.get("채용공고목록", []):
        공고번호 = item.get("공고번호", "no_id")
        채용제목 = item.get("채용제목", "")
        회사명 = item.get("회사명", "")
        근무지역 = item.get("근무지역", "")
        
        # 추가적으로 '상세정보' 등 더 많은 텍스트를 모아서 하나의 문자열로 합칠 수 있음
        text_block = f"제목: {채용제목}\n회사: {회사명}\n근무지: {근무지역}"
        
        # 필요하면 상세 정보나, 세부 요건들까지 붙여서 텍스트로 구성
        상세정보 = item.get("상세정보", {})
        # 예시: 상세 정보 내 string 항목만 추출
        for key, val in 상세정보.items():
            if isinstance(val, str):
                text_block += f"\n{val}"
            elif isinstance(val, list):
                joined_val = " ".join(map(str, val))
                text_block += f"\n{joined_val}"
        
        # LangChain의 Text Splitter로 chunk 나누기
        text_splits = text_splitter.split_text(text_block)
        
        # 각 청크를 Document 객체로 만들기
        for idx, chunk in enumerate(text_splits):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "공고번호": 공고번호,
                        "chunk_idx": idx
                    }
                )
            )
    return docs


def build_vectorstore(docs, persist_dir="./chroma_data", collection_name="job_postings"):
    """
    docs: List[Document]
    """
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    # 영구 저장 (persist_directory에 DuckDB/Parquet 형태로 기록)
    vectorstore.persist()
    return vectorstore

def load_vectorstore(persist_dir="./chroma_data", collection_name="job_postings"):
    """
    이미 저장된 VectorStore 불러오기
    """
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    return vectorstore


def main():
    # 1) JSON에서 문서 로드
    docs = load_and_prepare_documents("jobs.json")
    
    # 2) Chroma VectorStore 생성 및 저장
    vectorstore = build_vectorstore(docs)

    # 3) 검색 테스트
    query = "물리치료사"
    results = vectorstore.similarity_search(query, k=3)

    print("==== 검색 결과 ====")
    for i, r in enumerate(results, start=1):
        print(f"[Rank {i}]")
        print("내용:", r.page_content[:200], "...")
        print("메타데이터:", r.metadata)
        print("------------------------")


if __name__ == "__main__":
    main()
