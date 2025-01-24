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

# ========== 세션 관리를 위한 전역 상태 ==========
class ChatSession:
    def __init__(self):
        self.messages: List[Dict] = []
        self.user_profile: Dict = {
            "age": None,
            "location": None,
            "jobType": None,
            "experience": [],
            "preferred_jobs": [],
            "skills": [],
            "education": None,
            "job_status": None
        }

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
    
    def get_or_create_session(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession()
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, message: Dict, user_profile: Dict = None):
        session = self.get_or_create_session(session_id)
        session.messages.append(message)
        if user_profile:
            session.user_profile.update(user_profile)
    
    def get_messages(self, session_id: str) -> List[Dict]:
        session = self.get_or_create_session(session_id)
        return session.messages
    
    def get_user_profile(self, session_id: str) -> Dict:
        session = self.get_or_create_session(session_id)
        return session.user_profile

# 세션 매니저 인스턴스 생성
session_manager = SessionManager()

# ========== (B) 도구 정의 ==========
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
        info["age"] = int(age_pattern.group(1))
    
    # 지역 추출
    location_keywords = ["서울", "강남구", "강북구", "부산", "대구", "인천", "광주", "대전", "울산"]
    for loc in location_keywords:
        if loc in text:
            info["location"] = loc
    
    # 직종 추출
    job_keywords = {
        "강사": ["강사", "교사", "교육", "강의"],
        "경비": ["경비", "보안", "시설관리"],
        "운전": ["운전", "기사", "택시", "버스"],
        "사무": ["사무", "행정", "회계", "경리"]
    }
    
    for job_type, keywords in job_keywords.items():
        if any(keyword in text for keyword in keywords):
            info["jobType"] = job_type
            if "preferred_jobs" not in info:
                info["preferred_jobs"] = []
            info["preferred_jobs"].append(job_type)
    
    # 학력 추출
    education_keywords = {
        "대학교 졸업": ["대학교", "대졸"],
        "고등학교 졸업": ["고등학교", "고졸"],
        "중학교 졸업": ["중학교", "중졸"]
    }
    
    for edu_level, keywords in education_keywords.items():
        if any(keyword in text for keyword in keywords):
            info["education"] = edu_level
    
    # 구직 상태 추출
    status_keywords = {
        "구직중": ["구직", "구직중", "취업준비"],
        "재직중": ["재직", "재직중", "근무중"],
        "퇴직": ["퇴직", "은퇴"]
    }
    
    for status, keywords in status_keywords.items():
        if any(keyword in text for keyword in keywords):
            info["job_status"] = status
    
    # 기술/자격증 추출
    skill_keywords = {
        "컴퓨터 활용능력": ["컴퓨터", "컴활", "엑셀", "워드"],
        "전기기사": ["전기기사", "전기", "전기공사"],
        "운전면허": ["운전면허", "면허증"],
        "영어": ["영어", "토익", "토플"]
    }
    
    for skill, keywords in skill_keywords.items():
        if any(keyword in text for keyword in keywords):
            if "skills" not in info:
                info["skills"] = []
            info["skills"].append(skill)
    
    # 경력 추출
    if "경력 없음" in text or "신입" in text:
        info["experience"] = ["신입"]
    elif "경력" in text:
        experience_pattern = re.search(r"경력\s*(\d+)년", text)
        if experience_pattern:
            info["experience"] = [f"{experience_pattern.group(1)}년 경력"]
    
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

# ========== (C) LangChain 설정 ==========
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

# ========== (D) LangGraph 구성 ==========
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
            print(f"챗봇 노드 처리 중 오류: {str(e)}")
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

# ========== (E) API 엔드포인트 ==========
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

        # 세션 가져오기
        session = session_manager.get_or_create_session(request.session_id)
        
        # 메시지에서 사용자 정보 추출 및 프로필 업데이트
        info = extract_user_info_from_text(request.user_message)
        
        # 프로필 업데이트 (기존 정보 유지하면서 새로운 정보 추가)
        for key, value in info.items():
            if key in session.user_profile:
                if isinstance(session.user_profile[key], list):
                    if isinstance(value, list):
                        # 중복 제거하면서 리스트 확장
                        session.user_profile[key].extend([v for v in value if v not in session.user_profile[key]])
                    else:
                        if value not in session.user_profile[key]:
                            session.user_profile[key].append(value)
                else:
                    # 리스트가 아닌 경우 새 값으로 덮어쓰기
                    session.user_profile[key] = value
        
        # 프론트엔드에서 전달된 프로필 정보로 업데이트
        if request.user_profile:
            for key, value in request.user_profile.items():
                if value:  # 값이 있는 경우만 업데이트
                    session.user_profile[key] = value

        # 대화 컨텍스트 구성
        messages = []
        messages.append(SystemMessage(content=SYSTEM_MESSAGE))
        
        # 프로필 정보를 구조화된 형태로 추가
        profile_info = f"""현재 사용자 정보:
- 나이: {session.user_profile.get('age', '미입력')}세
- 거주지역: {session.user_profile.get('location', '미입력')}
- 희망직종: {session.user_profile.get('jobType', '미입력')}
- 경력: {', '.join(session.user_profile.get('experience', [])) if session.user_profile.get('experience') else '미입력'}
- 보유스킬/자격증: {', '.join(session.user_profile.get('skills', [])) if session.user_profile.get('skills') else '미입력'}
- 학력: {session.user_profile.get('education', '미입력')}
- 구직상태: {session.user_profile.get('job_status', '미입력')}"""
        
        messages.append(SystemMessage(content=profile_info))
        
        # 이전 대화 이력 추가 (최근 대화만)
        recent_messages = session.messages[-6:] if len(session.messages) > 6 else session.messages
        for msg in recent_messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # 현재 메시지 추가
        messages.append(HumanMessage(content=request.user_message))
        
        # 상태 생성 및 그래프 실행
        state = StateDict(
            messages=messages,
            user_profile=session.user_profile
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
            # 챗봇 응답을 세션에 저장
            session_manager.update_session(
                request.session_id,
                {"role": "assistant", "content": last_response}
            )
            logger.info(f"Sending response to session {request.session_id}: {last_response}")
        
        # 사용자 메시지 저장
        session_manager.update_session(
            request.session_id,
            {"role": "user", "content": request.user_message}
        )
        
        return {
            "responses": [last_response] if last_response else [],
            "user_profile": session.user_profile,
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
