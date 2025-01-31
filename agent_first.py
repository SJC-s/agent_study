import os
import re
from dotenv import load_dotenv
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from datetime import datetime

# =========================
# 1. LangChain, OpenAI, Tools
# =========================
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import BaseMessage

# =========================
# 2. LangGraph 관련
# =========================
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

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
        temperature=0.7,
        streaming=True
    )
    return llm

# ========== (D) LangGraph 구성 ==========

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: Dict

def build_graph():
    # 1) LLM 준비
    llm = setup_openai()
    tools = [search_jobs, update_user_profile]
    
    # LLM에 도구 바인딩
    llm_with_tools = llm.bind_tools(tools)

    # 2) 그래프 빌더 초기화
    graph_builder = StateGraph(State)

    # 3) 메인 챗봇 노드
    def chatbot_node(state: State):
        # 시스템 메시지와 사용자 프로필 정보 추가
        messages = [SystemMessage(content=SYSTEM_MESSAGE)]
        if "messages" in state and state["messages"]:
            messages.extend(state["messages"])
        
        # 프로필 정보를 문자열로 변환하여 컨텍스트에 추가
        if "user_profile" in state:
            profile_info = f"\n현재 사용자 정보:\n{str(state['user_profile'])}"
            messages.append(SystemMessage(content=profile_info))
        
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"챗봇 응답 생성 중 오류: {str(e)}")
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

# ========== (E) 콘솔 대화 실행 ==========

def run_console_chat():
    print("👋 고령자 취업 지원 AI 상담사입니다.")
    print("\n다음과 같은 도움을 드릴 수 있습니다:")
    print("- 경력과 경험을 고려한 맞춤형 일자리 추천")
    print("- 이력서와 자기소개서 작성 도움")
    print("- 고령자 특화 취업 정보 제공")
    print("- 면접 준비 도움")
    print("- 온라인 취업 사이트 활용 방법 안내")
    print("\n대화를 종료하시려면 '종료' 또는 'quit'를 입력해주세요.")

    try:
        # 그래프 생성
        graph = build_graph()
        
        # 사용자 프로필 초기화
        user_profile = UserProfile()
        
        # 설정
        config = {"configurable": {"thread_id": "demo-user"}}

        print("\nAI 상담사: 안녕하세요! 먼저 나이가 어떻게 되시나요?")

        while True:
            user_input = input("\n사용자: ").strip()
            
            if user_input.lower() in ["종료", "quit", "exit"]:
                print("\nAI 상담사: 상담을 종료합니다. 좋은 하루 되세요!")
                break
                
            if not user_input:
                print("메시지를 입력해주세요.")
                continue

            # 사용자 정보 추출 및 업데이트
            info = extract_user_info_from_text(user_input)
            for key, value in info.items():
                user_profile.update(key, value)

            try:
                # 그래프 실행
                events = graph.stream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "user_profile": user_profile.data
                    },
                    config,
                    stream_mode="values"
                )

                # 응답 처리
                for event in events:
                    if "messages" in event:
                        ai_msg = event["messages"][-1]
                        print(f"\nAI 상담사: {ai_msg.content}")
                        print("\n" + "-"*50)
            except Exception as e:
                print(f"\n오류가 발생했습니다: {str(e)}")
                print("다시 시도해주세요.")
                print("\n" + "-"*50)

    except Exception as e:
        print(f"시스템 초기화 중 오류가 발생했습니다: {str(e)}")
        print("프로그램을 종료합니다.")

if __name__ == "__main__":
    run_console_chat()
