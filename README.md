# 취업 도우미 AI 챗봇

이 프로젝트는 취업 준비생들을 위한 AI 챗봇으로, 이력서 작성, 자기소개서 작성, 면접 준비 등을 도와주는 서비스입니다.

## 주요 기능

- 이력서 작성 가이드 및 피드백
- 자기소개서 작성 조언
- 면접 준비 팁 제공
- 채용 트렌드 정보 제공
- 경력 개발 상담

## 설치 방법

1. 저장소를 클론합니다:
```bash
git clone [repository-url]
cd chatbot_aiagent
```

2. 필요한 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

3. `.env` 파일을 설정합니다:
- `.env` 파일을 열어 OpenAI API 키를 입력합니다:
```
OPENAI_API_KEY=your-api-key-here
```

## 실행 방법

다음 명령어로 애플리케이션을 실행합니다:
```bash
streamlit run app.py
```

## 사용 방법

1. 웹 브라우저에서 제공된 URL로 접속합니다 (기본적으로 http://localhost:8501)
2. 채팅창에 질문이나 요청사항을 입력합니다
3. AI 어시스턴트가 적절한 답변과 조언을 제공합니다

## 주의사항

- OpenAI API 키가 필요합니다
- API 사용량에 따라 비용이 발생할 수 있습니다 


검색 엔진용 tavily-python
Anthropic 모델 사용을 위한 langchain_anthropic
LangGraph 핵심 라이브러리