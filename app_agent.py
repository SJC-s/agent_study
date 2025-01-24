# app_agent.py

import os
import re
from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# =========================
# 기존 임포트
# =========================
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

import logging

# SQLAlchemy 임포트
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정: React 앱의 URL을 허용
origins = [
    "http://localhost:3000",  # React 앱 URL
    # 배포 시 다른 origins 추가
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 프론트엔드 URL로 업데이트
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# 데이터베이스 설정
# ========================

# 데이터베이스 연결 정보 가져오기
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB")

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

# SQLAlchemy 엔진 및 세션 설정
engine = create_engine(DATABASE_URL, pool_recycle=3600, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ORM 모델 정의
class ConversationLog(Base):
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True, nullable=False)
    role = Column(Enum('user', 'assistant', 'system'), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserProfile(Base):
    __tablename__ = "user_profiles"

    session_id = Column(String(255), primary_key=True, index=True)
    age = Column(Integer, nullable=True)
    location = Column(String(255), nullable=True)
    jobType = Column(String(255), nullable=True)
    experience = Column(JSON, nullable=True)
    preferred_jobs = Column(JSON, nullable=True)
    skills = Column(JSON, nullable=True)
    education = Column(String(255), nullable=True)
    job_status = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 테이블 생성 (필요시)
Base.metadata.create_all(bind=engine)

# ========================
# 세션 매니저 정의
# ========================

class ChatSession:
    def __init__(self, user_profile: Dict):
        self.user_profile = user_profile

class SessionManager:
    def __init__(self):
        pass

    def get_session_db(self, session_id: str):
        db = SessionLocal()
        try:
            user_profile = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
            if not user_profile:
                # 프로필이 없으면 새로 생성
                user_profile = UserProfile(session_id=session_id)
                db.add(user_profile)
                db.commit()
                db.refresh(user_profile)
            return user_profile
        except SQLAlchemyError as e:
            logger.error(f"DB 오류: {str(e)}")
            raise
        finally:
            db.close()

    def update_user_profile_db(self, session_id: str, user_profile: Dict):
        db = SessionLocal()
        try:
            profile = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
            if not profile:
                profile = UserProfile(session_id=session_id)
                db.add(profile)
            for key, value in user_profile.items():
                if hasattr(profile, key):
                    # 빈 문자열을 None으로 변환
                    if isinstance(value, str) and value.strip() == '':
                        setattr(profile, key, None)
                    else:
                        setattr(profile, key, value)
            db.commit()
        except SQLAlchemyError as e:
            logger.error(f"프로필 업데이트 오류: {str(e)}")
            raise
        finally:
            db.close()

    def get_messages_db(self, session_id: str) -> List[Dict]:
        db = SessionLocal()
        try:
            logs = db.query(ConversationLog).filter(ConversationLog.session_id == session_id).order_by(ConversationLog.created_at.asc()).all()
            messages = []
            for log in logs:
                messages.append({
                    "role": log.role,
                    "content": log.content
                })
            return messages
        except SQLAlchemyError as e:
            logger.error(f"메시지 조회 오류: {str(e)}")
            raise
        finally:
            db.close()

    def add_message_db(self, session_id: str, role: str, content: str):
        db = SessionLocal()
        try:
            log = ConversationLog(
                session_id=session_id,
                role=role,
                content=content
            )
            db.add(log)
            db.commit()
        except SQLAlchemyError as e:
            logger.error(f"메시지 추가 오류: {str(e)}")
            raise
        finally:
            db.close()

# 세션 매니저 인스턴스 생성
session_manager = SessionManager()

# =========================
# 도구 정의
# =========================

def fake_job_search(query: str) -> str:
    """
    고령자 맞춤 일자리 정보를 검색하는 가상 함수
    실제 구현 시 외부 API나 DB 연동 필요
    """
    jobs_db = {
        "경비": "아파트 경비직 - 월 220만원, 주 5일 근무",
        "운전": "마을버스 운전직 - 월 250만원, 탄력근무",
        "사무": "노인복지관 행정직 - 월 200만원, 주 5일",
        "강사": "실버복지관 강사직 - 시간당 2만원, 파트타임"
    }
    result = []
    for k, v in jobs_db.items():
        if k in query.lower():
            result.append(f"- {v}")
    return "\n".join(result) if result else "해당하는 일자리 정보가 없습니다."

@tool
def search_jobs(query: str) -> str:
    """고령자 맞춤 일자리를 검색하는 도구"""
    return fake_job_search(query)

def extract_user_info_from_text(text: str) -> dict:
    """대화 내용에서 사용자 정보를 추출"""
    info = {}

    # 나이 추출
    age_pattern = re.search(r"(\d+)[세살]", text)
    if age_pattern:
        age_value = age_pattern.group(1)
        info["age"] = int(age_value) if age_value.isdigit() else None

    # 지역 추출
    location_keywords = ["서울", "강남구", "강북구", "부산", "대구", "인천", "광주", "대전", "울산"]
    for loc in location_keywords:
        if loc in text:
            info["location"] = loc
            break  # 첫 번째 매칭되는 지역만 추출

    # 직종 추출
    job_keywords = {
        "jobType": ["강사", "교사", "교육", "강의", "경비", "보안", "시설관리", "운전", "기사", "택시", "버스", "사무", "행정", "회계", "경리"]
    }

    preferred_jobs = []
    for job_type, keywords in job_keywords.items():
        for keyword in keywords:
            if keyword in text:
                preferred_jobs.append(job_type)
                break
    if preferred_jobs:
        info["preferred_jobs"] = preferred_jobs
        # Assuming jobType is the primary job type
        info["jobType"] = preferred_jobs[0]

    # 학력 추출
    education_keywords = {
        "education": {
            "대학교 졸업": ["대학교", "대졸"],
            "고등학교 졸업": ["고등학교", "고졸"],
            "중학교 졸업": ["중학교", "중졸"]
        }
    }

    for edu_level, keywords in education_keywords["education"].items():
        if any(keyword in text for keyword in keywords):
            info["education"] = edu_level
            break

    # 구직 상태 추출
    status_keywords = {
        "job_status": {
            "구직중": ["구직", "구직중", "취업준비"],
            "재직중": ["재직", "재직중", "근무중"],
            "퇴직": ["퇴직", "은퇴"]
        }
    }

    for status, keywords in status_keywords["job_status"].items():
        if any(keyword in text for keyword in keywords):
            info["job_status"] = status
            break

    # 기술/자격증 추출
    skill_keywords = {
        "skills": {
            "컴퓨터 활용능력": ["컴퓨터", "컴활", "엑셀", "워드"],
            "전기기사": ["전기기사", "전기", "전기공사"],
            "운전면허": ["운전면허", "면허증"],
            "영어": ["영어", "토익", "토플"]
        }
    }

    skills = []
    for skill, keywords in skill_keywords["skills"].items():
        if any(keyword in text for keyword in keywords):
            skills.append(skill)
    if skills:
        info["skills"] = skills

    # 경력 추출
    if "경력 없음" in text or "신입" in text:
        info["experience"] = ["신입"]
    elif "경력" in text:
        experience_pattern = re.search(r"경력\s*(\d+)년", text)
        if experience_pattern:
            experience_years = experience_pattern.group(1)
            info["experience"] = [f"{experience_years}년 경력"]

    return info

@tool
def update_user_profile(text: str) -> str:
    """사용자 프로필 정보를 추출하고 업데이트하는 도구"""
    info = extract_user_info_from_text(text)
    return str(info)

@tool
def generate_resume_template(info: str) -> str:
    """
    사용자의 경력, 스킬, 희망 직종 등을 바탕으로 이력서 샘플(템플릿)을 반환하는 도구.
    실제로는 GPT를 추가적으로 호출하거나, 미리 저장된 템플릿을 가공할 수도 있습니다.
    """
    # 예시: 단순 문자열 리턴
    return f"""[이력서 템플릿]
이름:
나이:
경력:
보유 기술:
학력:
희망 직종:
직무 경력 사항:
"""

# =========================
# LangChain 설정
# =========================

SYSTEM_MESSAGE = """당신은 50세 이상 고령층의 취업을 돕는 AI 취업 상담사입니다.
사용자의 경험과 강점을 파악하여 맞춤형 일자리를 추천하고, 구직 활동을 지원합니다.

제공하는 주요 기능:
1. 경력/경험 기반 맞춤형 일자리 추천
2. 이력서 및 자기소개서 작성 가이드
3. 고령자 특화 취업 정보 제공
4. 면접 준비 및 커리어 상담
5. 디지털 취업 플랫폼 활용 방법 안내

상담 진행 방식:
1. 사용자의 기본 정보(나이, 경력, 희망 직종 등) 파악
2. 개인별 강점과 경험 분석
3. 맞춤형 일자리 정보 제공
4. 구체적인 취업 준비 지원

항상 다음 사항을 준수합니다:
- 쉽고 명확한 용어 사용
- 단계별로 상세한 설명 제공
- 공감과 이해를 바탕으로 한 응대
- 실질적이고 구체적인 조언 제시
- 이전 대화 내용을 기억하고 참조하여 응답"""

def setup_openai():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        streaming=True
    )
    return llm

# =========================
# LangGraph 구성
# =========================

from langgraph.graph import StateGraph, START, END

class StateDict(BaseModel):
    messages: List[BaseMessage]
    user_profile: Dict

def build_graph():
    # 1) LLM 준비
    llm = setup_openai()
    tools = [search_jobs, update_user_profile, generate_resume_template]

    # LLM에 도구 바인딩
    llm_with_tools = llm.bind_tools(tools)

    # 2) 그래프 빌더 초기화
    graph_builder = StateGraph(StateDict)

    # 3) 메인 챗봇 노드
    def chatbot_node(state: StateDict):
        try:
            # 시스템 메시지와 사용자 프로필 정보 추가
            messages = [SystemMessage(content=SYSTEM_MESSAGE)]

            # 기존 메시지 추가
            for msg in state.messages:
                if isinstance(msg, (HumanMessage, SystemMessage, AIMessage)):
                    messages.append(msg)
                elif isinstance(msg, dict) and 'content' in msg:
                    if msg.get('role') == 'user':
                        messages.append(HumanMessage(content=msg['content']))
                    elif msg.get('role') == 'assistant':
                        messages.append(AIMessage(content=msg['content']))
                    elif msg.get('role') == 'system':
                        messages.append(SystemMessage(content=msg['content']))

            # 프로필 정보를 문자열로 변환하여 컨텍스트에 추가
            if state.user_profile:
                profile_info = f"""현재 사용자 정보:
- 나이: {state.user_profile.get('age', '미입력')}세
- 거주지역: {state.user_profile.get('location', '미입력')}
- 희망직종: {state.user_profile.get('jobType', '미입력')}
- 경력: {', '.join(state.user_profile.get('experience', [])) if state.user_profile.get('experience') else '미입력'}
- 보유스킬/자격증: {', '.join(state.user_profile.get('skills', [])) if state.user_profile.get('skills') else '미입력'}
- 학력: {state.user_profile.get('education', '미입력')}
- 구직상태: {state.user_profile.get('job_status', '미입력')}"""
                messages.append(SystemMessage(content=profile_info))

            # LLM 호출
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        except Exception as e:
            logger.error(f"챗봇 노드 처리 중 오류: {str(e)}")
            return {"messages": [AIMessage(content="죄송합니다. 응답을 생성하는 중에 문제가 발생했습니다. 다시 한 번 말씀해 주시겠어요?")]}

    graph_builder.add_node("chatbot", chatbot_node)

    # 4) 도구 노드
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # 5) 조건부 라우팅
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    # 6) 체크포인터
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# 그래프 초기화
graph = build_graph()

# =========================
# API 엔드포인트 정의
# =========================

# FastAPI 요청 모델 정의
class ChatRequest(BaseModel):
    user_message: str
    user_profile: Dict = {}
    session_id: str

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    if not request.user_message:
        raise HTTPException(status_code=400, detail="사용자 메시지가 필요합니다.")

    try:
        logger.info(f"Received message from session {request.session_id}: {request.user_message}")

        # 세션 가져오기 (DB 기반)
        user_profile_record = session_manager.get_session_db(request.session_id)
        user_profile = {
            "age": user_profile_record.age,
            "location": user_profile_record.location,
            "jobType": user_profile_record.jobType,
            "experience": user_profile_record.experience or [],
            "preferred_jobs": user_profile_record.preferred_jobs or [],
            "skills": user_profile_record.skills or [],
            "education": user_profile_record.education,
            "job_status": user_profile_record.job_status
        }

        # 메시지에서 사용자 정보 추출 및 프로필 업데이트
        info = extract_user_info_from_text(request.user_message)

        # 프로필 업데이트 (DB에 저장)
        if info:
            session_manager.update_user_profile_db(request.session_id, info)
            # 업데이트된 프로필 다시 가져오기
            user_profile_record = session_manager.get_session_db(request.session_id)
            user_profile = {
                "age": user_profile_record.age,
                "location": user_profile_record.location,
                "jobType": user_profile_record.jobType,
                "experience": user_profile_record.experience or [],
                "preferred_jobs": user_profile_record.preferred_jobs or [],
                "skills": user_profile_record.skills or [],
                "education": user_profile_record.education,
                "job_status": user_profile_record.job_status
            }

        # 프론트엔드에서 전달된 프로필 정보로 업데이트
        if request.user_profile:
            # 빈 문자열을 None으로 변환
            sanitized_user_profile = {k: (v if v != '' else None) for k, v in request.user_profile.items()}
            session_manager.update_user_profile_db(request.session_id, sanitized_user_profile)
            # 업데이트된 프로필 다시 가져오기
            user_profile_record = session_manager.get_session_db(request.session_id)
            user_profile = {
                "age": user_profile_record.age,
                "location": user_profile_record.location,
                "jobType": user_profile_record.jobType,
                "experience": user_profile_record.experience or [],
                "preferred_jobs": user_profile_record.preferred_jobs or [],
                "skills": user_profile_record.skills or [],
                "education": user_profile_record.education,
                "job_status": user_profile_record.job_status
            }

        # 이전 대화 기록 가져오기
        recent_messages = session_manager.get_messages_db(request.session_id)

        # 대화 컨텍스트 구성
        messages = []
        messages.append(SystemMessage(content=SYSTEM_MESSAGE))

        # 프로필 정보를 구조화된 형태로 추가
        profile_info = f"""현재 사용자 정보:
- 나이: {user_profile.get('age', '미입력')}세
- 거주지역: {user_profile.get('location', '미입력')}
- 희망직종: {user_profile.get('jobType', '미입력')}
- 경력: {', '.join(user_profile.get('experience', [])) if user_profile.get('experience') else '미입력'}
- 보유스킬/자격증: {', '.join(user_profile.get('skills', [])) if user_profile.get('skills') else '미입력'}
- 학력: {user_profile.get('education', '미입력')}
- 구직상태: {user_profile.get('job_status', '미입력')}"""

        messages.append(SystemMessage(content=profile_info))

        # 이전 대화 이력 추가 (최근 대화만)
        recent_messages_to_include = recent_messages[-6:] if len(recent_messages) > 6 else recent_messages
        for msg in recent_messages_to_include:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))

        # 현재 메시지 추가
        messages.append(HumanMessage(content=request.user_message))

        # 상태 생성 및 그래프 실행
        state = StateDict(
            messages=messages,
            user_profile=user_profile
        )

        events = graph.stream(
            state.model_dump(),
            {"configurable": {"thread_id": request.session_id}},
            stream_mode="values"
        )

        # 응답 처리
        last_response = None
        for event in events:
            if "messages" in event and event["messages"]:
                for msg in event["messages"]:
                    if isinstance(msg, (AIMessage, SystemMessage)):
                        last_response = msg.content
                    elif isinstance(msg, dict) and 'content' in msg:
                        last_response = msg['content']

        if last_response:
            # 챗봇 응답을 DB에 저장
            session_manager.add_message_db(
                request.session_id,
                "assistant",
                last_response
            )
            logger.info(f"Sending response to session {request.session_id}: {last_response}")

        # 사용자 메시지 저장
        session_manager.add_message_db(
            request.session_id,
            "user",
            request.user_message
        )

        return {
            "responses": [last_response] if last_response else [],
            "user_profile": user_profile,
            "last_message": {
                "user": request.user_message,
                "assistant": last_response
            }
        }

    except Exception as e:
        logger.error(f"chat_endpoint 오류: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"메시지 처리 중 오류가 발생했습니다: {str(e)}"
        )

# 서버 시작 시 실행할 코드
@app.on_event("startup")
async def startup_event():
    # OpenAI API 키 확인
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    # 모델 초기화
    global graph
    graph = build_graph()
