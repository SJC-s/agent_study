import json
import re
import logging
import time
from typing import List
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 1. 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("build_vectorstore.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 2. 임베딩 모델 설정 (Sentence-BERT 계열)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 3. Text Splitter 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,  # 청크 크기 (문자 수 기준) 100~500, 50
    chunk_overlap=20  # 중첩 길이, chunk_size > chunk_overlap
)



# ----------------------- 커스텀 예시
# from langchain.text_splitter import TextSplitter

# class CustomKeywordTextSplitter(TextSplitter):
#     def __init__(self, keyword: str, chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.keyword = keyword
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap

#     def split_text(self, text: str) -> List[str]:
#         sections = text.split(self.keyword)
#         chunks = []
#         current_chunk = ""
        
#         for section in sections:
#             if len(current_chunk) + len(section) + len(self.keyword) > self.chunk_size:
#                 chunks.append(current_chunk.strip())
#                 current_chunk = self.keyword + section
#             else:
#                 current_chunk += self.keyword + section
        
#         if current_chunk:
#             chunks.append(current_chunk.strip())
        
#         # 중첩 처리
#         overlapped_chunks = []
#         for i in range(len(chunks)):
#             if i == 0:
#                 overlapped_chunks.append(chunks[i])
#             else:
#                 overlapped_chunks.append(chunks[i-1][-self.chunk_overlap:] + " " + chunks[i])
        
#         return overlapped_chunks

# # 3. Text Splitter 설정
# text_splitter = CustomKeywordTextSplitter(
#     keyword="공고내용",
#     chunk_size=1000,
#     chunk_overlap=200
# )






# 4. NER 파이프라인 설정 (Hugging Face Transformers)
# 존재하는 한국어 NER 모델로 변경 (예: monologg/koelectra-base-v3-finetuned-ner)
try:
    ner_pipeline = pipeline(
        "ner",
        model="Leo97/KoELECTRA-small-v3-modu-ner",
        tokenizer="Leo97/KoELECTRA-small-v3-modu-ner",
        aggregation_strategy="simple"
    )
    logger.info("NER 파이프라인이 성공적으로 초기화되었습니다.")
except Exception as e:
    logger.error(f"NER 모델 로드 실패: {e}")
    ner_pipeline = None  # NER 기능을 비활성화하거나 대체 방안을 고려

def load_data(json_file: str) -> dict:
    """
    JSON 파일을 로드하여 파이썬 딕셔너리로 반환.
    """
    logger.info(f"Loading data from {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data.get('채용공고목록', []))} job postings")
    return data

def clean_text(text: str) -> str:
    """
    텍스트 전처리: HTML 태그 제거 및 불필요한 공백 제거
    """
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    return text.strip()

def prepare_documents(data: dict) -> List[Document]:
    """
    JSON 데이터를 LangChain의 Document 객체 리스트로 변환.
    """
    logger.info("Preparing documents from data")
    documents = []
    for item in data.get("채용공고목록", []):
        공고번호 = item.get("공고번호", "no_id")
        채용제목 = clean_text(item.get("채용제목", ""))
        회사명 = clean_text(item.get("회사명", ""))
        근무지역 = clean_text(item.get("근무지역", ""))
        
        # 상세정보 텍스트 결합
        상세정보 = item.get("상세정보", {})
        상세정보_text = ""
        for key, val in 상세정보.items():
            if isinstance(val, str):
                상세정보_text += f"{val}\n"
            elif isinstance(val, list):
                상세정보_text += " ".join([clean_text(str(v)) for v in val if isinstance(v, str)]) + "\n"
        
        combined_text = f"제목: {채용제목}\n회사: {회사명}\n근무지: {근무지역}\n상세정보:\n{상세정보_text}"
        
        # 텍스트 분할
        splits = text_splitter.split_text(combined_text)
        
        # Document 객체 생성
        for idx, chunk in enumerate(splits):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "공고번호": 공고번호,
                        "채용제목": 채용제목,
                        "회사명": 회사명,
                        "근무지역": 근무지역,
                        "청크인덱스": idx
                    }
                )
            )
    logger.info(f"Prepared {len(documents)} document chunks")
    return documents

def build_vectorstore(documents: List[Document], persist_dir: str = "./chroma_data", collection_name: str = "job_postings") -> Chroma:
    """
    LangChain의 Chroma 벡터 스토어에 문서 추가 및 저장
    """
    logger.info("Building vector store")
    start_time = time.time()
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    
    vectorstore.persist()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Vector store built and persisted in {elapsed_time:.2f} seconds")
    logger.info(f"Vector store directory: {persist_dir}")
    return vectorstore

def main():
    json_file = "jobs.json"  # 실제 파일 경로로 변경
    
    # 1) 데이터 로드
    data = load_data(json_file)
    
    # 2) Document 리스트 준비
    docs = prepare_documents(data)
    
    # 3) 벡터 스토어 구축 및 저장 (처음 한 번만 실행)
    try:
        vectorstore = build_vectorstore(docs)
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        return

if __name__ == "__main__":
    main()
