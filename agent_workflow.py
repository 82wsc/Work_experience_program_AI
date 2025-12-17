import os
from typing import List, Dict, TypedDict, Union
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from rag_utils import query_chroma
from rag_utils_target import query_chroma_targeting

# Pydantic 모델 정의 (LLM의 구조화된 출력을 위해)
class Persona(BaseModel):
    target_group_index: int = Field(description="타겟 그룹의 순번")
    target_name: str = Field(description="타겟 세그먼트의 이름")
    target_features: str = Field(description="타겟 세그먼트의 주요 특징")
    classification_reason: str = Field(description="이 세그먼트를 분류한 데이터 기반의 근거")

class CampaignTitleResult(BaseModel):
    campaignTitle: str

class Personas(BaseModel):
    personas: List[Persona]

# 1. State 구현: CampaignState TypedDict
class CampaignState(TypedDict):
    """
    LangGraph의 상태를 정의하는 TypedDict.
    모든 Agent가 공유하는 중앙 데이터 구조입니다.
    """
    input_data: Dict # BE 서버로부터의 초기 요청 데이터 (예: core_benefit_text, custom_columns 등)
    target_personas: Union[List[Dict], None] # Targeting Agent의 타겟 5개 분류 결과
    messages_drafts: Union[List[Dict], None] # Messaging Agent의 타겟별 초안 2개 생성 결과
    validation_reports: Union[List[Dict], None] # Validator Agent의 초안 검증 리포트
    rework_count: int # 메시지 재생성 시도 횟수 (무한 루프 방지용)
    refine_feedback: Union[Dict, None] # 마케터의 재요청 피드백
    final_output: Union[Dict, None] # Formatter Agent의 최종 결과

# RAG Tool 구현
def rag_search(query: str, source_type: str) -> str:
    """
    RAG (Retrieval Augmented Generation) 툴입니다.
    query_chroma를 호출하여 Knowledge_Base DB 및 벡터 저장소에서 관련 지식을 검색하고,
    결과를 LLM 프롬프트에 포함하기 좋은 단일 문자열로 포맷팅합니다.

    Args:
        query (str): 검색할 쿼리.
        source_type (str): 검색할 지식의 출처 타입 (예: '정책', '성공 사례', '스팸/광고 정책').

    Returns:
        str: 검색된 관련 지식 요약 문자열.
    """
    print(f"RAG Search Called - Query: '{query}', Source Type: '{source_type}'")
    
    # rag_utils의 query_chroma 함수를 사용하여 ChromaDB에서 검색
    search_results = query_chroma(
        query_texts=[query],
        n_results=3, # 관련성 높은 3개 결과 사용
        where_filter={"source_type": source_type}
    )
    
    if not search_results:
        return "관련 지식을 찾을 수 없습니다."
    
    # 검색 결과를 단일 문자열로 포맷팅
    formatted_knowledge = "\n".join([
        f"- {result['document']} (출처: {result['metadata'].get('title', 'N/A')}, 관련성 점수: {1-result['distance']:.2f})"
        for result in search_results
    ])
    
    return f"'{source_type}' 관련 검색된 지식:\n{formatted_knowledge}"


def rag_search_targeting(query: str) -> str:
    """
    소비자 세그먼트 논문 DB용 RAG 검색.
    """
    print(f"Targeting RAG Search Called - Query: '{query}'")

    results = query_chroma_targeting(
        query_texts=[query],
        n_results=5,
        where_filter=None  # 논문 메타데이터 필터 필요 시 추가 가능
    )

    if not results:
        return "관련 세그먼트 지식을 찾을 수 없습니다."

    formatted = "\n".join([
        f"- {r['document']} (출처: {r['metadata'].get('title', 'N/A')}, 점수: {1-r['distance']:.2f})"
        for r in results
    ])

    return f"[세그먼트 관련 지식]\n{formatted}"


# 2. Agent 함수 구현 (LLM 연동)

# LLM, Parser, Prompt 등 공통 컴포넌트 초기화
# 참고: OpenAI API 키는 환경변수 'OPENAI_API_KEY'에 설정되어 있어야 합니다.
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
json_parser = JsonOutputParser()

def run_targeting_agent(state: CampaignState) -> Dict:
    """
    Targeting Agent: 마케터의 핵심 혜택을 기반으로 5개의 상이한 타겟 페르소나를 분류합니다.
    상태에 이미 페르소나가 존재하면, 해당 페르소나를 그대로 사용합니다.
    """
    print("---" + " Targeting Agent 실행 중 ---")
    
    # 상태에 이미 페르소나가 존재하면, 해당 페르소나를 그대로 사용하고 다음 단계로 넘어갑니다.
    if state.get('target_personas'):
        print("기존 페르소나를 재사용합니다.")
        return {"target_personas": state['target_personas']}

    # Pydantic 모델을 사용하는 JSON 파서 초기화
    pydantic_parser = JsonOutputParser(pydantic_object=Personas)

    input_data = state.get('input_data', {})
    core_benefit_text = input_data.get('coreBenefitText', '기본 혜택')
    refine_feedback = state.get('refine_feedback', None)
    custom_columns = input_data.get('customColumns', {})

    if isinstance(custom_columns, dict):
        formatted_columns = "\n".join([f"- {k}: {v}" for k, v in custom_columns.items()])
    else:
        formatted_columns = str(custom_columns)

    # 소비자 세그먼트 논문 기반 RAG
    segment_knowledge = rag_search_targeting(
        query="소비자 세그먼트 분류 기준 및 소비 패턴별 그룹 특징"
    )

    # RAG Tool 호출: 정책 관련 지식 검색
    policy_knowledge = rag_search(query=f"{core_benefit_text} 관련 정책", source_type='정책')
    print(f"Targeting Agent - RAG Knowledge: {policy_knowledge}")

    # LLM 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 KT의 전문 마케팅 분석가입니다. 
            아래 세 가지 정보를 기반으로 5개의 서로 다른 타겟 세그먼트를 도출해야 합니다:

            1) 프로모션 핵심 혜택  
            2) 마케터가 제공한 customColumns (고객 DB의 Feature)  
            3) 소비자 세그먼트 논문 기반 RAG 지식  

            **중요 규칙**
            - 각 세그먼트는 반드시 customColumns 중 최소 1개 이상을 기반으로 해야 합니다.
            - 논문 기반 소비 패턴 / 세그먼트 기준을 반드시 반영해야 합니다.
            - 현실적인 고객 DB 세그멘테이션 규칙(구매 빈도, 나이, 선호 카테고리 등)을 반영해야 합니다.
            - 단순 페르소나가 아니라 **데이터 기반 세그먼트 그룹**을 출력해야 합니다.
            - 모든 출력 필드(target_name, target_features, classification_reason)는 반드시 한국어로 작성해야 합니다.
            - 마케터 수정 피드백은 세그먼트 내용을 구성하는 데에만 참고하고, JSON 출력 형식은 반드시 유지해야 합니다.

            {format_instructions}
            """),

            ("human", """
            프로모션 핵심 혜택:
            {core_benefit}

            마케터 수정 피드백:
            {refine_feedback_text}

            사용 가능한 고객 데이터 컬럼(customColumns):
            {custom_columns}

            소비자 세그먼트 관련 RAG 지식:
            {segment_knowledge}

            프로모션 정책 관련 RAG 지식:
            {policy_knowledge}

            위 정보를 기반으로 5개의 데이터 기반 타겟 세그먼트를 생성해주세요.
            """)
        ]).partial(format_instructions=pydantic_parser.get_format_instructions())

    # LangChain Expression Language (LCEL) 체인 구성
    chain = prompt | llm | pydantic_parser

    # 체인 실행
    response_dict = chain.invoke({
        "core_benefit": core_benefit_text,
        "refine_feedback_text": refine_feedback.get('details', '없음') if refine_feedback else '없음',
        "custom_columns": formatted_columns,
        "segment_knowledge": segment_knowledge,
        "policy_knowledge": policy_knowledge
    })

    # Pydantic 파서는 이미 딕셔너리를 반환합니다.
    target_personas = response_dict.get("personas", [])
    print(f"Targeting Agent - 생성된 타겟 페르소나: {target_personas}")
    return {"target_personas": target_personas}

def summarize_target_features(target_features: str) -> str:
    prompt = f"""
아래 타겟 특징을 바탕으로,
마케팅 메시지에 자연스럽게 들어갈 수 있는
짧은 대상 규정 표현을 작성하세요.

규칙:
- 반드시 한 문장
- 문장은 반드시 '사람이라면' 으로 끝나야 한다
- '당신이라면', '고객이라면', '분이라면', 너라면 등의 단어 사용 금지
- 조건 나열 금지 (AND 구조 금지)
- 구체적 서비스명, 수치, 브랜드명 금지
- 데이터/분류/전문 용어 금지
- 20자 내외로 간결하게

타겟 특징:
{target_features}
"""
    result = llm.invoke(prompt)
    return result.content.strip()



def generate_campaign_title(core_benefit_text: str) -> str:
    prompt = f"""
    아래 핵심 혜택 설명을 읽고,
    소비자에게 보여줄 수 있는 '프로모션 이름' 하나를 생성하세요.

    규칙:
    - 반드시 명사형
    - 하나의 이벤트명처럼 간결
    - "~을 안내하는", "~를 위한", "프로모션" 금지
    - 혜택의 성격이 드러나야 함
    - 15자 내외 권장

    혜택 설명:
    {core_benefit_text}

    JSON 형식:
    {{
      "campaignTitle": "생성된 제목"
    }}
    """

    parser = JsonOutputParser()
    chain = llm | parser
    result = chain.invoke(prompt)
    return result["campaignTitle"]


def run_messaging_agent(state: CampaignState) -> Dict:
    print("--- Messaging Agent 실행 중 ---")

    input_data = state.get("input_data", {})
    target_personas = state.get("target_personas", [])
    rework_count = state.get("rework_count", 0)
    validation_reports = state.get("validation_reports")
    refine_feedback = state.get("refine_feedback")

    core_benefit_text = input_data.get("coreBenefitText", "기본 혜택")
    campaign_title = input_data.get("campaignTitle")

    if not campaign_title:
        campaign_title = generate_campaign_title(core_benefit_text)

    # custom columns
    custom_columns_data = input_data.get("customColumns", {})
    if isinstance(custom_columns_data, dict):
        columns_for_prompt = "\n".join([f"- `{{{k}}}`: ({v})" for k, v in custom_columns_data.items()])
    else:
        columns_for_prompt = ", ".join(custom_columns_data)

    # source urls
    source_urls = input_data.get("sourceUrls", [])
    source_urls_str = ", ".join(source_urls) if source_urls else "없음"

    
    # ----------------------------
    # 이름 컬럼 존재 여부 판단 (customColumns 기준)
    # ----------------------------
    custom_columns_data = input_data.get("customColumns", {})

    name_column_exists = False

    if isinstance(custom_columns_data, dict):
        for col_name in custom_columns_data.keys():
            normalized = col_name.lower().replace(" ", "")
            if (
                "이름" in col_name
                or "고객명" in col_name
                or "name" in normalized
            ):
                name_column_exists = True
                break

    # 공통 prompt
    prompt = ChatPromptTemplate.from_messages([
    ("system", """
name_column_exists: {name_column_exists}

당신은 고객 데이터와 프로모션 정보를 바탕으로
서로 다른 톤의 마케팅 메시지 2개를 생성하는 전문 카피라이터입니다.

반드시 아래 규칙을 100% 준수하십시오.

---

## 이름 치환 규칙 (중요)

- name_column_exists가 true인 경우에만 아래 치환 규칙을 적용한다.
- name_column_exists가 false인 경우에는 기존 표현을 그대로 사용한다.

[치환 규칙 – name_column_exists = true]
- "고객님, 당신" → "[이름]님"
- "너" → "[이름]"

[기본 규칙 – name_column_exists = false]
- "고객님", "너" 표현을 그대로 유지한다.

추가 규칙:
- "[이름]"은 실제 이름이 아닌 표시용 토큰이다.
- "[이름]" 외의 이름 표현은 절대 생성하지 않는다.
- 한 문장에 "[이름]" 토큰은 최대 1회만 사용한다.

---

## 공통 입력 변수
- {coreBenefitText}
- {target_name}
- {target_features}
- {source_urls}
- {feedback_instructions}
- {target_features_summary}
- {campaignTitle}
---
        
## 출력 결과물
- 메시지 초안은 정확히 2개
- JSON 형식으로만 출력
- 각 message_text는 반드시 하나의 완성된 메시지 문단

---

## 초안 1: 세련·우아 스타일 가이드 메시지 (고정 무드)

⚠️ 중요  
초안 1은 **어떤 프로모션이든 아래 구조와 세련된 분위기를 반드시 유지**해야 한다.  
다만, **문장 표현은 허용된 변주 규칙 범위 내에서만 유연하게 변경 가능**하다.  
문단 순서, 전체 흐름, 감정선, 말투는 절대 변경하지 않는다.

### 초안 1 작성 규칙

1. 광고 문구처럼 보이는 표현 금지
2. “최대”, “파격”, “놓치지 마세요” 등 직접적인 행동 유도 표현 지양
3. 혜택은 나열보다 **문장 중심의 서술**
4. 감성 키워드 사용 허용  
   (무드, 안목, 선택, 분위기, 일상, 취향, 품격 등)


⚠️ 줄바꿈 및 레이아웃 규칙 (필수)

- 각 문단은 반드시 줄바꿈(개행)으로 분리하여 출력한다.
- 인사 문단, 출시 안내 문단, 혜택 소개 문단, 혜택 블록, 공감 문단, 마무리 문단은
  각각 하나의 독립된 문단이어야 한다.
- 서로 다른 문단을 한 줄로 합치거나 이어서 출력하는 행위는 금지한다.
- 혜택 블록의 각 혜택 줄은 반드시 줄바꿈으로 구분되어야 한다.
- 개행이 없는 출력은 구조 위반으로 간주한다.

---

### 문체 변주 규칙 (중요)

아래 항목에 한해 **의미와 감정선은 유지하되, 문장 표현의 다양화를 허용**한다.

- 인사 문장: 정중하고 부드러운 표현 범위 내에서 자연스럽게 변주 가능
- 도입부 문장: “소식을 전합니다”, “안내드립니다” 등의 표현은 의미를 유지한 채 변형 가능
- 혜택 소개 연결 문장: 의미는 유지하되 문장 표현은 자유롭게 조정 가능
- 마무리 문장: 여운과 정중함을 유지하는 범위 내에서 표현 변주 가능
- 동일한 문장이 반복되지 않도록 자연스럽게 표현을 변경할 것

단, 아래는 절대 변경하지 않는다.
- 전체 문단 순서
- 차분하고 세련된 문체
- 과장되지 않은 우아한 톤

---

### 초안 1 고정 템플릿 (구조 고정 / 문장 표현 유연)

[정중한 인사 및 메시지 도입 문장 – 세련되고 차분한 어조로 시작]

[평온한 분위기와 연결된 안내 문장 – 일상 속 작은 여유와 만족을 암시]

이번 "{campaignTitle}"은  
당신의 일상에 조금 더 여유롭고 만족스러운 선택지를 더하기 위해 마련되었습니다.  
필요할 때, 충분하게. 만족스러운 선택을 중요하게 생각하는 분들을 위한 구성입니다.

이번 기회를 통해 만나보실 수 있는 주요 혜택은 다음과 같습니다.

[혜택_블록_시작]

아래 {coreBenefitText}에는 번호, 불릿(-), 하위 항목으로 구성된 혜택 정보가 포함되어 있다.

[혜택 분해 규칙]
- {coreBenefitText}에 등장하는 모든 혜택 정보는 분해 대상이다.
- 상위 불릿이 하위 항목 목록을 소개하는 설명 역할일 경우,
  해당 상위 불릿은 독립 혜택으로 출력하지 않는다.
- 하위 항목들은 반드시 하나의 혜택 문장으로 통합하여 출력한다.
- 동일한 의미의 혜택이 두 줄 이상 출력되면 실패로 간주한다.
- 어떤 혜택도 생략해서는 안 된다.

[출력 규칙]
- 각 혜택은 반드시 한 줄로 출력한다.
- 각 줄은 반드시 이모티콘 1개로 시작해야 한다.
- 이모티콘 앞에는 어떤 문자도 오면 안 된다.
- 각 줄은 하나의 독립된 혜택이며 한 문장만 허용한다.
- 줄글, 문단 묶기, 혜택 병합은 중대한 규칙 위반이다.

[출력 형식 예시]
🎬 혜택 설명 한 문장
🛍️ 혜택 설명 한 문장
📱 혜택 설명 한 문장

이제 위 규칙에 따라  
{coreBenefitText}에 포함된 모든 혜택을 빠짐없이 출력하시오.  

[혜택_블록_종료]

이러한 혜택은 고객님처럼 {target_features_summary} 충분히 만족하실겁니다.

필요한 순간에 부담 없이 선택하실 수 있도록 준비한 이번 프로모션이  
당신의 일상에 작은 만족으로 남길 바랍니다.

{coreBenefitText}에서 확인 가능한 기간 내 제공

👉 자세히 보기: {source_urls}

---


## 초안 2: 고정 캐주얼 프로모션 템플릿 (강제)

⚠️ 중요  
초안 2는 **어떤 프로모션이든 아래 구조와 캐주얼한 분위기를 반드시 유지**해야 한다.  
다만, **문장 표현은 허용된 변주 규칙 범위 내에서만 유연하게 변경 가능**하다.  
문단 순서, 전체 흐름, 감정선은 절대 변경하지 않는다.

### 초안 2 작성 규칙

1. 반드시 인사 + 가벼운 대화체로 시작
2. 전체 메시지는 친구에게 말하듯 자연스러운 구어체
3. 느낌표, 이모지, 감탄 표현 사용 허용
4. 혜택은 리스트 형식으로 나열
5. 마지막은 행동 유도 + 가벼운 여운 멘트로 종료
6. 타겟 특성은 **직접 설명하지 말고**, 말투와 상황 예시 속에 자연스럽게 녹일 것
     

⚠️ 줄바꿈 및 레이아웃 규칙 (필수)

- 각 문단은 반드시 줄바꿈(개행)으로 분리하여 출력한다.
- 인사 문단, 출시 안내 문단, 혜택 소개 문단, 혜택 블록, 공감 문단, 마무리 문단은
  각각 하나의 독립된 문단이어야 한다.
- 서로 다른 문단을 한 줄로 합치거나 이어서 출력하는 행위는 금지한다.
- 혜택 블록의 각 혜택 줄은 반드시 줄바꿈으로 구분되어야 한다.
- 개행이 없는 출력은 구조 위반으로 간주한다.

---

### 초안 2 고정 템플릿 (구조·톤 고정)

아래 템플릿의 문단 순서와 전체 분위기는 유지하되,  
문장 표현은 아래 변주 규칙 범위 내에서 유연하게 조정할 수 있습니다.

⚠️ 문체 변주 규칙 (중요)

아래 항목에 한해 표현의 다양화를 허용한다.
의미와 감정선은 유지하되, 문장 표현은 매번 달라질 수 있다.

- 인사 문장: 친근한 인사와 호기임 유도 표현 2~3개 중 자연스럽게 선택
- 설렘 표현: "두근두근", "기다리던", "드디어" 중 일부를 생략하거나 교체 가능
- 혜택 소개 연결 문장: 의미는 유지하되 문장 표현은 자유롭게 변형 가능
- 행동 유도 문장: 동일한 의미 내에서 다른 구어체 표현 사용 가능

단, 아래는 절대 변경하지 않는다.
- 전체 문단 순서
- 캐주얼하고 친근한 말투
- 친구에게 말하듯 하는 대화체 톤
     
---

### 초안 2 고정 템플릿 (구조 고정 / 문장 표현 유연)

[인사 및 호기심 유도 문장 – 활기찬 인사와 설레는 호기심 유도]

[설렘을 담은 출시 안내 문장 – "{campaignTitle}" 출시 소식을 친근하고 활기차게 전달]

이번 프로모션, 그냥 지나치기엔 너무 아깝거든!  
어떤 혜택이 있는지 가볍게 정리해 줄게 👀

[혜택_블록_시작]

아래 {coreBenefitText}에는  
번호, 불릿(-), 하위 항목으로 구성된 혜택 정보가 포함되어 있습니다.

[혜택 분해 및 통합 규칙]
- {coreBenefitText}에 포함된 모든 혜택 정보는 반드시 분해 대상입니다.
- 상위 항목이 하위 목록을 소개하는 설명 역할만 할 경우,
  해당 상위 항목은 단독 혜택으로 출력하지 않습니다.
- 하위 항목들은 의미를 통합하여 하나의 혜택 문장으로 작성합니다.
- 동일하거나 유사한 의미의 혜택이 중복 출력되면 실패로 간주합니다.
- 어떤 혜택도 누락해서는 안 됩니다.

[혜택 출력 규칙]
- 각 혜택은 반드시 한 줄로 출력합니다.
- 각 줄은 반드시 이모티콘 1개로 시작합니다.
- 이모티콘 앞에는 어떠한 문자도 허용되지 않습니다.
- 각 혜택 출력시 존댓말을 사용하지 않고 친근하면서도 가벼운 말투를 사용합니다.
- 줄글, 문단 병합, 혜택 묶기는 허용되지 않습니다.

[출력 예시 형식]
🎬 혜택 설명 한 문장
🛍️ 혜택 설명 한 문장
📱 혜택 설명 한 문장

위 규칙에 따라  
{coreBenefitText}에 포함된 모든 혜택을 빠짐없이 출력하십시오.

[혜택_블록_종료]
         
이런 혜택은 너처럼 {target_features_summary} 절대 그냥 지나칠 수 없을걸?

괜히 “좀 더 일찍 볼 걸” 싶어질 수도 있으니까  
시간 있을 때 한 번만 슬쩍 확인해 봐 😉

{coreBenefitText}에서 확인 가능한 기간 내 제공

👉 자세히 보기: {source_urls}

---

## 최종 출력 형식 (엄수)

{{
  "drafts": [
    {{
      "message_draft_index": 1,
      "message_text": "초안 1 메시지 전문"
    }},
    {{
      "message_draft_index": 2,
      "message_text": "초안 2 고정 캐주얼 템플릿 적용 메시지 전문"
    }}
  ]
}}

---

이 규칙을 어기면 출력은 실패로 간주됩니다.""")
]).partial(
    name_column_exists="true" if name_column_exists else "false"
)

    chain = prompt | llm | json_parser

    # ----------------------------------------------------
    # 1) refine_feedback 있으면 → 전체 재작성
    # ----------------------------------------------------
    if refine_feedback:
        print("--- 실행 모드: MarKeTer refine 전체 재작성 ---")

        messages_drafts = []
        feedback_instructions = "마케터 피드백을 반영해 전면 재작성하세요."
        feedback_section = refine_feedback.get("details", "")

        for persona in target_personas:
            persona_features = persona["target_features"]
            persona_features_summary = summarize_target_features(persona_features)
            response = chain.invoke({
                "campaignTitle": campaign_title,
                "coreBenefitText": core_benefit_text,
                "source_urls": source_urls_str,
                "feedback_instructions": feedback_instructions,
                "feedback_section": feedback_section,
                "target_name": persona["target_name"],
                "target_features": persona["target_features"],
                "target_features_summary": persona_features_summary,
                "columns": columns_for_prompt,
            })

            messages_drafts.append({
                "target_group_index": persona["target_group_index"],
                "target_name": persona["target_name"],
                "message_drafts": response.get("drafts", []),
            })

        return {"messages_drafts": messages_drafts, "rework_count": 0}

    # ----------------------------------------------------
    # 2) validation_reports FAIL 포함 → 부분 재작성
    # ----------------------------------------------------
    if validation_reports:
        print("--- 실행 모드: Validation 기반 재작성 판단 ---")

        personas_to_rework = set()
        feedback_per_persona = {}

        for report in validation_reports:
            if report.get("policy_compliance") == "FAIL" or report.get("spam_risk_score", 0) > 70:
                idx = report["target_group_index"]
                personas_to_rework.add(idx)
                if idx not in feedback_per_persona:
                    feedback_per_persona[idx] = []
                feedback_per_persona[idx].append(report.get("recommended_action", ""))

        if personas_to_rework:
            print(f"부분 재작성 대상: {personas_to_rework}")

            messages_drafts = []
            for persona in target_personas:
                persona_features = persona["target_features"]
                persona_features_summary = summarize_target_features(persona_features)
                group_idx = persona["target_group_index"]

                if group_idx in personas_to_rework:
                    all_feedback = "\n".join(feedback_per_persona[group_idx])
                    feedback_instr = "검증 실패 항목을 기준으로 메시지를 재작성하세요."

                    response = chain.invoke({
                        "campaignTitle": campaign_title,
                        "coreBenefitText": core_benefit_text,
                        "source_urls": source_urls_str,
                        "feedback_instructions": feedback_instr,
                        "feedback_section": all_feedback,
                        "target_name": persona["target_name"],
                        "target_features": persona["target_features"],
                        "target_features_summary": persona_features_summary,
                        "columns": columns_for_prompt,
                    })

                    messages_drafts.append({
                        "target_group_index": group_idx,
                        "target_name": persona["target_name"],
                        "message_drafts": response.get("drafts", []),
                    })
                else:
                    # 기존 유지
                    existing = next(
                        (d for d in state["messages_drafts"] if d["target_group_index"] == group_idx),
                        None
                    )
                    if existing:
                        messages_drafts.append(existing)

            return {"messages_drafts": messages_drafts, "rework_count": rework_count + 1}

    # ----------------------------------------------------
    # 3) 초기 메시지 생성
    # ----------------------------------------------------
    print("--- 실행 모드: 초기 메시지 생성 ---")

    messages_drafts = []
    for persona in target_personas:
        persona_features = persona["target_features"]
        persona_features_summary = summarize_target_features(persona_features)       
        response = chain.invoke({
            "campaignTitle": campaign_title,
            "coreBenefitText": core_benefit_text,
            "source_urls": source_urls_str,
            "feedback_instructions": "",
            "feedback_section": "",
            "target_name": persona["target_name"],
            "target_features": persona["target_features"],
            "target_features_summary": persona_features_summary,
            "columns": columns_for_prompt,
        })

        messages_drafts.append({
            "target_group_index": persona["target_group_index"],
            "target_name": persona["target_name"],
            "message_drafts": response.get("drafts", []),
        })

    return {"messages_drafts": messages_drafts, "rework_count": rework_count}

def run_validator_agent(state: CampaignState) -> Dict:
    """
    Validator Agent: 생성된 메시지 초안을 검증하고, 필요한 경우 피드백을 제공합니다.
    """
    print("---" + " Validator Agent 실행 중 ---")
    messages_drafts = state.get('messages_drafts', [])
    core_benefit_text = state.get('input_data', {}).get('coreBenefitText', '')

    # LLM 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 메시지 검토 및 법규 준수 전문가입니다. 당신의 임무는 주어진 메시지 초안을 아래 3가지 관점에서
        **엄격하게 평가**하고 구조화된 JSON 리포트를 작성하는 것입니다.

        1.  **스팸 위험도 (0~100점):** 과도한 이모티콘, 특수문자, 긴급성 강조 문구 사용 여부. 점수가 높을수록 위험.
        2.  **정보의 정확성/정책 준수:** RAG 지식 기반으로 혜택 조건 등이 사실과 일치하는지 확인.
        3.  **개선 의견:** 실제 발송 전 수정이 필요한 부분을 명확히 제시.

        결과는 반드시 아래 JSON 형식의 단일 객체로 반환해야 합니다.
        'policy_compliance'가 'FAIL'일 경우, 'review_summary'는 반드시 "위반 사유: [인용문]" 으로 시작해야 하며, RAG 지식에서 위반된 정책의 핵심 내용을 정확히 인용해야 합니다.
        {{
            "spam_risk_score": <0-100 사이의 정수>,
            "policy_compliance": "<'PASS' 또는 'FAIL'>",
            "review_summary": "<(FAIL 시) 위반 사유: [인용문]을 포함한 검토 요약>",
            "recommended_action": "<구체적인 개선 제안 또는 '없음'>"
        }}
        """),
        ("human", """
        검토할 메시지 초안:
        ---
        {message_text}
        ---
        
        프로모션 핵심 혜택: {core_benefit}
        참고용 RAG 지식 (스팸/광고 정책): {rag_knowledge}

        위 정보를 바탕으로 메시지 초안을 평가하고 JSON 리포트를 작성해주세요.
        """)
    ])

    # LangChain Expression Language (LCEL) 체인 구성
    chain = prompt | llm | json_parser

    validation_reports = []

    # RAG Tool 호출: 스팸/광고 정책을 한 번만 검색
    spam_policy_knowledge = rag_search(query="메시지 스팸/광고 정책", source_type='스팸/광고 정책')
    print(f"Validator Agent - RAG Knowledge for validation: {spam_policy_knowledge}")

    for target_group_drafts in messages_drafts:
        target_name = target_group_drafts['target_name']
        for draft in target_group_drafts['message_drafts']:
            message_text = draft['message_text']

            # 체인 실행
            report = chain.invoke({
                "message_text": message_text,
                "core_benefit": core_benefit_text,
                "rag_knowledge": spam_policy_knowledge
            })

            # 전체 리포트 저장
            report['target_group_index'] = target_group_drafts['target_group_index']
            report['message_draft_index'] = draft['message_draft_index']
            validation_reports.append(report)

    print(f"Validator Agent - 생성된 검증 리포트: {validation_reports}")

    # 에이전트는 이제 리포트만 반환하고, 재작업 결정은 decide_next_step에서 처리합니다.
    # 이전 피드백 상태를 확실히 지우기 위해 validator_feedback을 None으로 설정합니다.
    return {"validation_reports": validation_reports, "validator_feedback": None}


def run_formatter_agent(state: CampaignState) -> Dict:
    """
    Formatter Agent: 최종 결과를 통합하여 BE 서버로 전달할 JSON 형태로 포맷팅합니다.
    이 버전에서는 타겟 페르소나, 메시지 초안, 검증 리포트를 모두 결합합니다.
    """
    print("---" + " Formatter Agent 실행 중 ---")
    target_personas = state.get('target_personas', [])
    messages_drafts = state.get('messages_drafts', [])
    validation_reports = state.get('validation_reports', [])

    # 빠른 조회를 위해 리포트와 초안을 맵으로 변환합니다.
    report_map = {}
    if validation_reports:
        for report in validation_reports:
            key = (report['target_group_index'], report['message_draft_index'])
            report_map[key] = report

    draft_map = {}
    if messages_drafts:
        for group in messages_drafts:
            draft_map[group['target_group_index']] = group['message_drafts']

    # 페르소나를 기준으로 초안과 검증 리포트를 결합합니다.
    final_target_groups = []
    if target_personas:
        for persona in target_personas:
            group_index = persona['target_group_index']
            drafts_for_group = draft_map.get(group_index, [])
            
            new_drafts = []
            for draft in drafts_for_group:
                key = (group_index, draft['message_draft_index'])
                report_for_draft = report_map.get(key)
                
                new_draft_entry = {
                    "message_draft_index": draft['message_draft_index'],
                    "message_text": draft['message_text'],
                    "validation_report": report_for_draft
                }
                new_drafts.append(new_draft_entry)
            
            final_target_groups.append({
                "target_group_index": group_index,
                "target_name": persona['target_name'],
                "target_features": persona['target_features'],
                "classification_reason": persona.get('classification_reason', 'N/A'), # 이유 필드 추가
                "message_drafts": new_drafts
            })

    print(f"Formatter Agent - 최종 결합 결과: {final_target_groups}")
    return {"final_output": final_target_groups}

# 3. LangGraph 조건부 루프: decide_next_step 함수
def decide_next_step(state: CampaignState) -> str:
    """
    Validator 노드 이후 다음 단계를 결정합니다.
    재시도 횟수 및 검증 결과에 따라 'messaging' 노드로 루프백하거나 'formatter' 노드로 종료됩니다.
    """
    print("---" + " decide_next_step 실행 중 ---")
    rework_count = state.get('rework_count', 0)
    validation_reports = state.get('validation_reports', [])

    # 최대 재시도 횟수 (1회) 초과 시 강제 종료
    if rework_count >= 1:
        print(f"재시도 횟수 {rework_count}회 초과. Formatter로 이동하여 강제 종료.")
        return "formatter"

    # validation_reports를 직접 검사하여 재작업 필요 여부 확인
    needs_rework = False
    if validation_reports:
        for report in validation_reports:
            if report.get('policy_compliance') == 'FAIL' or report.get('spam_risk_score', 0) > 70:
                needs_rework = True
                break  # 하나라도 실패하면 즉시 재작업 결정

    if needs_rework:
        print(f"검증 실패. Messaging Agent로 루프백하여 메시지 재생성 시도. 현재 재시도 횟수: {rework_count}")
        return "messaging"
    else:
        print("모든 검증 통과. Formatter로 이동하여 최종 결과 포맷팅.")
        return "formatter"

# LangGraph 워크플로우 빌드
def build_agent_workflow():
    workflow = StateGraph(CampaignState)

    # 노드 추가
    workflow.add_node("targeting", run_targeting_agent)
    workflow.add_node("messaging", run_messaging_agent)
    workflow.add_node("validator", run_validator_agent)
    workflow.add_node("formatter", run_formatter_agent)

    # 시작점 설정 (분기 가능하도록)
    # 기본 시작점은 'targeting'
    workflow.set_entry_point("targeting") 
    # 'messaging'을 또 다른 진입점으로 설정
    # workflow.add_entry_point("messaging") # Removed as it causes an error

    # 엣지 연결
    workflow.add_edge("targeting", "messaging")
    workflow.add_edge("messaging", "validator")

    # 조건부 엣지 연결
    workflow.add_conditional_edges(
        "validator",
        decide_next_step,
        {
            "messaging": "messaging", # 재작업 필요 시 messaging 노드로 루프백
            "formatter": "formatter"  # 검증 성공 또는 재시도 횟수 초과 시 formatter 노드로
        }
    )

    # 종료 엣지
    workflow.add_edge("formatter", END)

    app = workflow.compile(checkpointer=None)
    return app

# 워크플로우 테스트 (선택 사항)
if __name__ == "__main__":
    app = build_agent_workflow()

    initial_state = {
        "input_data": {
            "coreBenefitText": "KT 5G 프리미엄 요금제, 데이터 완전 무제한!",
            "message_tone": "전문적이고 친근한",
            "custom_columns": ["[이름]", "[핸드폰기종]", "[사용년도]"]
        },
        "rework_count": 0,
        "target_personas": None,
        "messages_drafts": None,
        "validation_reports": None,
        "validator_feedback": None,
        "refine_feedback": None
    }

    print("---" + " LangGraph 워크플로우 시작 ---")
    # 스트리밍 방식으로 실행 결과를 확인합니다.
    for s in app.stream(initial_state):
        print(s)
        print("---")
    print("---" + " LangGraph 워크플로우 종료 ---")
