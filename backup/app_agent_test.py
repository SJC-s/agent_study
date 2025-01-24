import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

def console_chat():
    # OpenAI 기반 LLM 설정
    llm = ChatOpenAI(temperature=0.2)
    
    # 필요한 도구(tools) 로드
    # - 검색, 계산 등 다른 기능이 필요하다면 여기에 추가
    # - 예시: load_tools(["serpapi"], llm=llm)
    tools = []

    # LangChain Agent 초기화
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    print("콘솔 기반 챗봇에 오신 것을 환영합니다!")
    print("질문을 입력해보세요. '종료' 혹은 'quit'을 입력하면 대화를 끝냅니다.")

    while True:
        user_input = input("사용자: ")
        
        # 종료 조건
        if user_input.lower() in ["종료", "quit", "exit"]:
            print("대화를 종료합니다.")
            break
        
        # 에이전트에게 사용자 입력 전달 -> 답변 생성
        response = agent.run(user_input)
        
        # 답변 출력
        print("AI: " + response)

if __name__ == "__main__":
    console_chat()
