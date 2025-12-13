import os
import json
from typing import Dict, Any, Tuple, List, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import uuid
from rag_utils import query_chroma

# --- Pydantic 모델 정의 (LLM의 구조화된 출력을 위해) ---
class CampaignData(BaseModel):
    campaignTitle: str = Field(..., description="AI가 생성하거나 사용자가 수정한 캠페인 제목")
    coreBenefitText: str = Field(..., description="캠페인의 핵심 혜택")
    customColumns: Dict[str, Union[List[str], str]] = Field(..., description="고객 데이터 컬럼. 값은 카테고리형(리스트) 또는 서술형(문자열)일 수 있음.")
    sourceUrls: List[str] = Field(..., description="관련 URL 목록")

class AIResponse(BaseModel):
    """LLM이 반환해야 하는 JSON 구조"""
    next_ai_response: str = Field(description="사용자에게 보여줄 다음 AI의 응답 메시지")
    updated_campaign_data: CampaignData = Field(description="사용자 메시지를 반영하여 업데이트된 캠페인 데이터")
    is_ready_to_send: bool = Field(description="모든 정보가 수집되어 백엔드 서버로 전송할 준비가 되었는지 여부")

# --- 세션 데이터 구조 ---
def create_new_session_data() -> Dict[str, Any]:
    """새로운 대화 세션 데이터를 초기화합니다."""
    return {
        "conversation_id": str(uuid.uuid4()),
        "conversation_history": [],
        "campaign_data": {
            "campaignTitle": "",
            "coreBenefitText": "",
            "customColumns": {},
            "sourceUrls": [],
        },
    }

# --- LLM 및 Parser 초기화 ---
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))
json_parser = JsonOutputParser(pydantic_object=AIResponse)

# --- 메인 프롬프트 템플릿 ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
        당신은 사용자와 대화하며 마케팅 캠페인 생성을 위한 JSON 데이터를 구축하는 친절하고 유능한 마케팅 전문가 AI입니다.
        사용자는 데이터 구조를 전혀 모른다고 가정하고, '카테고리형', '서술형', '키', '값', 'JSON', '전송' 등 기술적이거나 혼란을 주는 용어를 절대 사용하지 마세요.

        ## 당신의 임무
        1.  마케터 동료와 대화하듯이, 한 번에 하나씩 질문을 주고받으며 자연스럽게 대화를 리드합니다.
        2.  대화 내용을 바탕으로 아래 'JSON 데이터 구조'를 단계적으로 완성합니다.
        3.  사용자의 입력을 간결하게 확인하고 (예: '알겠습니다.', '좋네요.'), 자연스럽게 다음 단계로 넘어갑니다.
        4.  모든 필수 정보가 채워지면, 사용자에게 최종 확인을 받고 대화를 마무리합니다.

        ## JSON 데이터 구조 (내부 참고용)
        - campaignTitle: AI가 생성하거나 사용자가 수정한 캠페인 제목
        - coreBenefitText: 캠페인의 핵심 혜택 (문자열)
        - customColumns: 고객 정보 객체.
            - 예시 1: 키 "등급"에 대한 값은 ["VIP", "Gold", "일반"] 처럼 리스트 형태여야 합니다. (AI가 카테고리형으로 판단한 경우)
            - 예시 2: 키 "가입일"에 대한 값은 "YYYY-MM-DD 형식" 처럼 문자열 형태여야 합니다. (AI가 서술형으로 판단한 경우)
        - sourceUrls: 관련 URL 주소 목록 (리스트)


        ## 대화 규칙
        -   **의도 파악 우선**: 당신의 최우선 임무는 정해진 순서대로 질문하는 것이 아니라, 사용자의 최근 메시지 의도를 파악하고 그에 맞게 행동하는 것입니다. 사용자는 아래의 의도를 가질 수 있습니다.
            1.  `초기 대화 시작/캠페인 제목 제공`: 사용자가 대화를 시작하거나 캠페인 제목을 제공합니다. (캠페인 제목이 아직 설정되지 않은 경우 우선적으로 처리)
            2.  `핵심 혜택 제공/수정`: 사용자가 캠페인의 핵심 혜택에 대해 말합니다.
            3.  `고객 정보 추가/수정`: 사용자가 타겟 고객의 정보(예: 등급, 연령)에 대해 말합니다.
            4.  `URL 제공/수정`: 사용자가 관련 링크에 대해 말합니다.
            5.  `상담 요청`: 사용자가 캠페인에 대한 조언이나 아이디어를 구합니다. (예: "어떤 프로모션을 더 할까요?")
            6.  `데이터 수집 완료/종료 요청`: 사용자가 정보 제공을 마치고 캠페인 생성을 마무리하고 싶어 합니다. (예: "이정도로 만들래", "여기까지 할래", "준비해줘", "이제 끝내고싶어", "그만할래", "나머지 내용은 내가 채워볼게")
            7.  `최종 생성 승인`: 당신의 "마무리할까요?" 질문에 사용자가 긍정적으로 답변합니다. (예: "네", "좋아요", "마무리해줘", "생성해줘")
            8.  `최종 생성 보류/수정`: 당신의 "마무리할까요?" 질문에 사용자가 부정적으로 답변하거나 수정을 원합니다. (예: "아니요", "잠깐만요", "수정할게")

        -   **의도에 따른 행동 지침**:
            -   **초기 대화 시작/캠페인 제목 제공 시**:
                - 만약 `current_campaign_data.campaignTitle`이 비어있다면, 사용자 메시지에서 가장 적절한 캠페인 제목을 추출하여 `updated_campaign_data.campaignTitle`에 반영하거나, 사용자에게 **"캠페인 제목을 무엇으로 할까요?"** 라고 직접 물어보세요.
                - 제목이 설정되면 **"'<updated_campaign_data.campaignTitle>' 이름으로 프로모션 생성을 시작하겠습니다. 이제 캠페인의 핵심 혜택이 무엇인지 알려주시겠어요?"** 라고 물어보며 핵심 혜택으로 유도합니다.
            -   **핵심 혜택 제공/수정 시**:
                - `updated_campaign_data.coreBenefitText`에 내용을 반영합니다.
                - 만약 `current_campaign_data.campaignTitle`이 여전히 비어있다면, `updated_campaign_data.campaignTitle` = `updated_campaign_data.coreBenefitText`를 바탕으로 제목을 생성하고 사용자에게 **"'<updated_campaign_data.campaignTitle>' 이름으로 프로모션 생성을 시작하겠습니다."** 라고 알려줍니다.
                - "내용을 추가했습니다. 혹시 혜택에 대해 덧붙일 내용이 더 있으신가요?" 라고 물어보세요. 사용자가 없다고 하면 다음 단계를 제안하면 됩니다.
            -   **고객 정보 추가/수정 시**:
                -   사용자가 '등급', '학년', '상태'처럼 명확한 구분값이 있는 정보를 언급하면, AI가 스스로 카테고리형 데이터라고 판단해야 합니다. **절대로 사용자에게 유형을 묻지 마세요.**
                -   그 다음, **"좋네요! '등급'은 보통 어떻게 나뉘나요? 쉼표(,)로 구분해서 알려주세요. (예: VIP, Gold, 일반)"** 처럼 사용자에게 친절하게 되물어서 실제 값(value)들을 리스트로 받아 `customColumns`에 추가하세요.
                -   사용자가 '관심사'처럼 자유로운 답변이 가능한 정보를 말하면, 서술형으로 판단하고 "네, '관심사' 항목을 추가했습니다. 어떤 내용이 들어갈지 간단한 설명을 덧붙여 주시겠어요? (예: '자유롭게 입력')" 와 같이 가이드를 주세요.
            -   **URL 제공/수정 시**: `sourceUrls`에 정보를 추가하거나 수정하세요. 그 후 "링크를 추가했습니다. 또 다른 링크가 있나요?" 라고 물어보세요.
            -   **상담 요청 시**: '참고 지식'과 현재 캠페인 데이터를 바탕으로 전문가적인 답변을 먼저 제공하세요.
            -   **데이터 수집 완료/종료 요청 시**: 사용자가 데이터 수집 마무리를 원하거나, 모든 필수 정보(핵심 혜택)가 자연스럽게 채워졌다고 판단되면, **"알겠습니다. 그럼 현재까지의 내용으로 데이터를 정리하겠습니다. 이 내용으로 캠페인 생성을 마무리할까요?"** 와 같이 최종 확인 질문을 하세요. **이때 `is_ready_to_send`는 항상 `false`입니다.**
            -   **최종 생성 승인 시**: 사용자가 최종 확인 질문에 **긍정적으로 응답하면**, `is_ready_to_send`를 `true`로 설정하고, `next_ai_response`에 "알겠습니다. 캠페인 데이터 생성이 완료되었습니다. 언제든지 다시 찾아주세요!" 와 같은 최종 완료 메시지를 담아주세요.
            -   **최종 생성 보류/수정 시**: 사용자가 최종 확인 질문에 **부정적으로 응답하면**, `is_ready_to_send`를 `false`로 유지하고, `next_ai_response`에 **"알겠습니다. 어떤 부분을 변경하고 싶으신가요? 편하게 말씀해주세요."** 와 같이 사용자가 수정을 계속할 수 있도록 친절하게 유도하는 질문을 담아주세요. **"전송을 취소했습니다" 와 같은 부정적이거나 혼란을 주는 메시지는 절대 보내면 안 됩니다.**

        -   **대화 가이드**: 사용자의 의도를 처리한 후, 또는 의도가 불분명할 경우, AI가 대화를 이끌어야 합니다.
            -   예시: "혜택 내용을 업데이트했습니다. 다음으로 프로모션을 진행하기 위해 어떤 고객 정보를 사용할 수 있을까요? 아니면 캠페인 혜택, 고객 정보, URL 등을 설정하거나 수정하고 싶으신가요?"
            -   예시: "무엇을 도와드릴까요? 캠페인 혜택, 고객 정보, URL 등을 설정하거나 수정할 수 있습니다."

        ## 현재 대화 상태
        -   **현재까지 완성된 JSON:** {current_campaign_data}
        -   **최근 대화 내용:** {conversation_history}
        -   **사용자의 최근 메시지:** {user_message}
        -   **참고 지식 (사용자가 질문한 경우에만 사용):** {rag_knowledge}

        ## 당신의 다음 행동
        위 대화 상태를 참고하여, 'JSON 데이터 구조'를 완성하기 위한 다음 질문을 생성해주세요.
        그리고 사용자 메시지를 반영하여 업데이트된 JSON 데이터와 함께, 아래 출력 형식에 맞춰 응답을 생성해주세요.
        어떤 설명도 덧붙이지 말고, 오직 JSON 객체만 반환해야 합니다.
        {format_instructions}
        """),
    ]
).partial(format_instructions=json_parser.get_format_instructions())

# --- LangChain 체인 구성 ---
chain = prompt_template | llm | json_parser

is_question_chain = (
    ChatPromptTemplate.from_template("다음 사용자 메시지가 조언, 정보, 분석을 요청하는 '질문'입니까? 오직 '네' 또는 '아니오'로만 답하세요.\n\n{user_message}")
    | llm
)

def process_message_with_llm(session_data: Dict[str, Any], user_message: str) -> Tuple[str, bool]:
    """
    LLM을 사용하여 대화 상태와 사용자 메시지를 기반으로 다음 AI 응답과 상태를 결정합니다.
    """
    
    # 1. 사용자의 메시지가 질문인지 판단
    is_question_response = is_question_chain.invoke({"user_message": user_message})
    is_question = "네" in is_question_response.content
    
    rag_knowledge = ""
    if is_question:
        print(f"DEBUG: User message detected as a question. Performing RAG search.")
        # 2. 질문일 경우 RAG 검색 수행
        try:
            search_results = query_chroma(query_texts=[user_message], n_results=3)
            if search_results:
                formatted_knowledge = "\n".join([f"- {result['document']}" for result in search_results])
                rag_knowledge = f"관련 정보:\n{formatted_knowledge}"
        except Exception as e:
            print(f"Error during RAG search: {e}")
            rag_knowledge = "관련 정보를 검색하는 중 오류가 발생했습니다."

    # 3. 메인 LLM 호출
    response_payload = chain.invoke({
        "current_campaign_data": json.dumps(session_data["campaign_data"], ensure_ascii=False),
        "conversation_history": "\n".join(f"- {h['role']}: {h['content']}" for h in session_data["conversation_history"][-4:]),
        "user_message": user_message,
        "rag_knowledge": rag_knowledge
    })

    # 4. 세션 데이터 업데이트
    session_data["campaign_data"] = response_payload["updated_campaign_data"]
    session_data["conversation_history"].append({"role": "user", "content": user_message})
    session_data["conversation_history"].append({"role": "assistant", "content": response_payload["next_ai_response"]})
    
    ai_response_text = response_payload["next_ai_response"]
    # LLM이 is_ready_to_send를 true로 설정하면, 이는 대화가 완전히 끝났음을 의미합니다.
    is_finished = response_payload["is_ready_to_send"]

    return ai_response_text, is_finished