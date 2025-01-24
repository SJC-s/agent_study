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

# ========== (A) 사용자 프로필 관리 ==========
class UserProfile:
    def __init__(self):
        self.data = {
            "age": None,
            "experience": [],
            "preferred_jobs": [],
            "skills": [],
            "location": None,
            "education": None,
            "job_status": None
        }
        self.conversation_state = "initial"
        self.last_update = datetime.now()

    def update(self, key: str, value: any):
        self.data[key] = value
        self.last_update = datetime.now()

    def get_profile_summary(self) -> str:
        if not self.data["age"]:
            return "프로필 정보가 아직 없습니다."
        
        summary = f"""현재 프로필 정보:
- 나이: {self.data['age']}세
- 선호 직종: {', '.join(self.data['preferred_jobs']) if self.data['preferred_jobs'] else '미입력'}
- 경력: {', '.join(self.data['experience']) if self.data['experience'] else '미입력'}
- 보유 기술: {', '.join(self.data['skills']) if self.data['skills'] else '미입력'}
- 희망 근무지: {self.data['location'] if self.data['location'] else '미입력'}
- 학력: {self.data['education'] if self.data['education'] else '미입력'}
- 현재 상태: {self.data['job_status'] if self.data['job_status'] else '미입력'}"""
        return summary

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
    
    # 직종 키워드 추출
    job_keywords = ["경비", "운전", "사무", "강사"]
    for job in job_keywords:
        if job in text:
            if "preferred_jobs" not in info:
                info["preferred_jobs"] = []
            info["preferred_jobs"].append(job)
    
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
- 실질적이고 구체적인 조언 제시"""

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
            if hasattr(state, 'messages') and state.messages:
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
            if hasattr(state, 'user_profile') and state.user_profile:
                profile_info = f"\n현재 사용자 정보:\n{str(state.user_profile)}"
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
    user_message: str  # message -> user_message로 변경
    user_profile: Dict = {
        "age": None,
        "location": None,
        "jobType": None,  # 프론트엔드의 필드명과 일치
        "experience": [],
        "preferred_jobs": [],
        "skills": [],
        "education": None,
        "job_status": None
    }

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    if not request.user_message:
        raise HTTPException(status_code=400, detail="사용자 메시지가 필요합니다.")
    
    try:
        # 사용자 프로필 초기화 또는 업데이트
        user_profile = request.user_profile
        
        # 메시지에서 사용자 정보 추출 및 프로필 업데이트
        info = extract_user_info_from_text(request.user_message)
        for key, value in info.items():
            if key in user_profile:
                if isinstance(user_profile[key], list):
                    if isinstance(value, list):
                        user_profile[key].extend(value)
                    else:
                        user_profile[key].append(value)
                else:
                    user_profile[key] = value

        # jobType을 preferred_jobs에 추가
        if user_profile.get("jobType"):
            if "preferred_jobs" not in user_profile:
                user_profile["preferred_jobs"] = []
            if user_profile["jobType"] not in user_profile["preferred_jobs"]:
                user_profile["preferred_jobs"].append(user_profile["jobType"])

        # 상태 업데이트
        state = StateDict(
            messages=[HumanMessage(content=request.user_message)],
            user_profile=user_profile
        )

        # 그래프 실행
        events = graph.stream(
            state.model_dump(),  # dict() 대신 model_dump() 사용
            {"configurable": {"thread_id": "demo-user"}},
            stream_mode="values"
        )
        
        # 응답 수집
        responses = []
        for event in events:
            if "messages" in event and event["messages"]:
                for msg in event["messages"]:
                    if isinstance(msg, (AIMessage, SystemMessage)):
                        responses.append(msg.content)
                    elif isinstance(msg, dict) and 'content' in msg:
                        responses.append(msg['content'])
        
        return {
            "responses": responses,
            "user_profile": user_profile
        }
    
    except Exception as e:
        print(f"chat_endpoint 오류: {str(e)}")
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
