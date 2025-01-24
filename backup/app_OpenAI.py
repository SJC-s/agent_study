import os
import getpass
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# .env 파일 로드
load_dotenv()

# OPENAI_API_KEY 환경 변수 설정
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API 키를 입력하세요: ")

# langchain_openai에서 ChatOpenAI 임포트
from langchain_openai import ChatOpenAI

# 유효한 모델 이름으로 설정 (예: "gpt-4o-mini)
model = ChatOpenAI(model="gpt-4o-mini")


# 시스템 메시지 정의
SYSTEM_MESSAGE = """당신은 채용 정보와 이력서 작성을 도와주는 전문 AI 어시스턴트입니다.
다음과 같은 도움을 제공할 수 있습니다:
1. 이력서 작성 조언과 피드백
2. 자기소개서 작성 가이드
3. 면접 준비 팁
4. 채용 트렌드 정보
5. 경력 개발 조언

항상 친절하고 전문적인 조언을 제공하며, 구체적인 예시와 함께 설명해주세요."""

def initialize_messages() -> List:
    """대화 이력을 초기화하는 함수"""
    return [SystemMessage(content=SYSTEM_MESSAGE)]

def get_ai_response(messages: List) -> str:
    """AI 응답을 생성하는 함수"""
    try:
        response = model(messages)
        return response.content
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

def main():
    print("💼 취업 도우미 AI 챗봇")
    print("\n안녕하세요! 취업 준비를 도와드리는 AI 어시스턴트입니다.")
    print("다음과 같은 도움을 드릴 수 있습니다:")
    print("- 이력서 작성 가이드 및 피드백")
    print("- 자기소개서 작성 조언")
    print("- 면접 준비 팁")
    print("- 채용 트렌드 정보")
    print("- 경력 개발 상담")
    print("\n종료하려면 'quit' 또는 'exit'를 입력하세요.\n")

    # 대화 이력 초기화
    messages = initialize_messages()

    while True:
        user_input = input("\n질문하기 >> ").strip()
        
        if user_input.lower() in ["quit", "exit"]:
            print("\n채팅을 종료합니다. 좋은 하루 되세요!")
            break
            
        if not user_input:
            print("메시지를 입력해주세요.")
            continue

        # 사용자 메시지 추가
        messages.append(HumanMessage(content=user_input))

        # AI 응답 생성
        ai_response = get_ai_response(messages)
        
        # AI 응답 출력
        print("\nAI 응답:")
        print(ai_response)
        print("\n" + "-"*50)
        
        # AI 응답을 대화 이력에 추가
        messages.append(AIMessage(content=ai_response))

if __name__ == "__main__":
    main()
