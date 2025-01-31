import json
import re
import logging
import time
import os
from typing import List
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # ChatOpenAI 사용
from langchain.chains import LLMChain
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 1. 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("search_and_summarize.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 2. 임베딩 모델 설정 (Sentence-BERT 계열)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 3. NER 파이프라인 설정 (Hugging Face Transformers)
#    모델이 실제 존재하는지 확인한 뒤 교체하세요.
NER_MODEL_NAME = "Leo97/KoELECTRA-small-v3-modu-ner"
try:
    logger.info(f"Attempting to load NER model: {NER_MODEL_NAME}")
    ner_pipeline = pipeline(
        "ner",
        model=NER_MODEL_NAME,
        tokenizer=NER_MODEL_NAME,
        aggregation_strategy="simple"
    )
    logger.info("NER 파이프라인이 성공적으로 초기화되었습니다.")
except Exception as e:
    logger.warning(f"NER 모델 로드 실패: {e}")
    ner_pipeline = None  # NER만 비활성화

# 4. 프롬프트 템플릿 설정
prompt_template = """
다음은 채용 공고 검색 결과입니다. 각 공고에 대한 요약과 주요 정보를 제공해 주세요.

검색 질의: {query}

검색 결과:
{results}

요약:
"""

prompt = PromptTemplate(
    input_variables=["query", "results"],
    template=prompt_template
)

def setup_chatopenai():
    """
    ChatOpenAI를 사용하여 v1/chat/completions 엔드포인트로 요청.
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. (.env 파일 확인)")

    # ChatOpenAI: chat 모델에 올바른 엔드포인트 사용 (gpt-3.5-turbo, gpt-4 등)
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",  # 실제 사용 모델 (gpt-4도 가능)
        temperature=0.7,
        streaming=False
    )
    return llm

# LLMChain 초기화
try:
    llm = setup_chatopenai()
    chain = LLMChain(llm=llm, prompt=prompt)
    logger.info("LLMChain이 성공적으로 초기화되었습니다.")
except Exception as e:
    logger.error(f"LLM 초기화 실패: {e}")
    chain = None

def load_vectorstore(persist_dir: str = "./chroma_data", collection_name: str = "job_postings") -> Chroma:
    """
    저장된 Chroma 벡터 스토어 로드
    """
    logger.info("Loading existing vector store")
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    logger.info("Vector store loaded successfully")
    return vectorstore

def search_documents(vectorstore: Chroma, query: str, top_k: int = 3) -> List[Document]:
    """
    질의를 벡터화하여 유사한 문서 검색
    """
    logger.info(f"Searching for query: '{query}' with top_k={top_k}")
    start_time = time.time()
    
    results = vectorstore.similarity_search(query, k=top_k)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Search completed in {elapsed_time:.4f} seconds")
    
    if not results:
        logger.warning("No search results found")
    else:
        logger.info(f"Found {len(results)} results")
    
    return results

def extract_entities(text: str) -> dict:
    """
    텍스트에서 개체명 추출 (Hugging Face Transformers 이용)
    """
    if not ner_pipeline:
        # NER 파이프라인이 없을 시, 빈 dict 반환
        return {}
    
    ner_results = ner_pipeline(text)
    entities = {}
    for ent in ner_results:
        label = ent['entity_group']
        if label not in entities:
            entities[label] = []
        entities[label].append(ent['word'])
    return entities

def display_entities(search_results: List[Document]):
    """
    검색 결과에서 개체명 추출 및 출력
    """
    print("\n==== 추출된 개체명 ====")
    for doc in search_results:
        ents = extract_entities(doc.page_content)
        print(f"공고번호: {doc.metadata.get('공고번호')}")
        print(f"추출된 개체명: {ents}")
        print("------------------------")

def generate_summary(query: str, search_results: List[Document]) -> str:
    """
    검색 결과를 요약하여 자연스러운 문장으로 생성
    """
    if not chain:
        logger.warning("LLMChain이 초기화되지 않았습니다.")
        return "요약을 생성할 수 없습니다."
    
    # 검색 결과를 간단히 문자열로 합침
    results_text = "\n".join([
        f"Rank {i+1}: {doc.page_content[:200]}..."
        for i, doc in enumerate(search_results)
    ])

    # chain.invoke: Dict 형태로 입력
    # DeprecationWarning: chain.run → chain.invoke
    try:
        response = chain.invoke({
            "query": query,
            "results": results_text
        })
        summary = response  # invoke 결과가 곧 요약문
    except Exception as e:
        logger.error(f"요약 생성 중 오류 발생: {e}")
        summary = "요약 생성에 실패했습니다."
    return summary

def main():
    # 1) 벡터 스토어 로드
    try:
        vectorstore = load_vectorstore()
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return
    
    # 2) 검색 질의 입력
    query = "경기도 운전기사 "  # 예시 질의, 필요에 따라 변경
    
    # 3) 문서 검색
    try:
        search_results = search_documents(vectorstore, query, top_k=3)
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return
    
    # 4) 검색 결과 출력
    print("\n==== 검색 결과 ====")
    for i, doc in enumerate(search_results, start=1):
        print(f"[Rank {i}]")
        print(f"내용: {doc.page_content[:200]} ...")
        print(f"메타데이터: {doc.metadata}")
        print("------------------------")
    
    # 5) NER 결과 출력
    display_entities(search_results)
    
    # 6) 프롬프트 엔지니어링을 통한 요약
    summary = generate_summary(query, search_results)
    print("\n==== 요약된 검색 결과 ====")
    print(summary)

if __name__ == "__main__":
    main()
