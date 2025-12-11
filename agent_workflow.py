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

def run_messaging_agent(state: CampaignState) -> Dict:
    print("--- Messaging Agent 실행 중 ---")

    input_data = state.get("input_data", {})
    target_personas = state.get("target_personas", [])
    rework_count = state.get("rework_count", 0)
    validation_reports = state.get("validation_reports")
    refine_feedback = state.get("refine_feedback")

    core_benefit_text = input_data.get("coreBenefitText", "기본 혜택")

    # custom columns
    custom_columns_data = input_data.get("customColumns", {})
    if isinstance(custom_columns_data, dict):
        columns_for_prompt = "\n".join([f"- `{{{k}}}`: ({v})" for k, v in custom_columns_data.items()])
    else:
        columns_for_prompt = ", ".join(custom_columns_data)

    # source urls
    source_urls = input_data.get("sourceUrls", [])
    source_urls_str = ", ".join(source_urls) if source_urls else "없음"

    # 공통 prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 고객 데이터를 기반으로 브랜드 톤에 맞는 마케팅 메시지를 설계하는 전문 카피라이터입니다.
아래의 3단계 프로세스를 엄격히 따라,
P-Type(Professional, 차분하고 신뢰감 있는 톤) 메시지 1개와
H-Type(Human, 친근하고 대화형 톤) 메시지 1개를 생성해야 합니다.

---

**[1단계: 분석 및 전략 수립]**

먼저, 주어진 모든 정보(페르소나, 핵심 혜택, RAG 지식)를 종합적으로 분석하고,  
각 초안에 대한 생성 전략을 머릿속으로 구체적으로 수립합니다.

아래 <생각 예시>는 당신의 사고 과정을 돕기 위한 참고 자료일 뿐,  
**이 내용을 그대로 모방하거나 실제 생성 메시지에 사용해서는 안 됩니다.**

<생각 예시>

페르소나 분석: 타겟은 ‘합리적 판단을 중시하지만, 일상적 소통 방식에도 민감한 30대 직장인’.
업무 중 정보를 빠르게 파악하길 원하고, 부담스럽지 않은 대화체 톤을 선호함.

RAG 지식 분석: 과거 성공 사례에서 이 그룹은

명확한 절차·조건 안내(P-Type 특징)에 높은 신뢰를 보였고

일상 언어 기반의 가벼운 공감 표현(H-Type 특징)에 긍정 반응을 보임.
반면, 과장 문구나 지나친 감성 표현은 신뢰도를 떨어뜨린 사례로 나타남.

초안 1 (P-Type) 전략:

공식 안내 톤 유지

조건·기준·이용 절차 중 핵심 1개를 명확히 제시

감정 표현·과장 금지, 안정적인 문장 구조 사용

초안 2 (H-Type) 전략:

고객의 상황을 가볍게 짚는 공감 문장으로 시작

자연스러운 대화체 흐름 유지

부담 없는 표현으로 혜택의 의미를 전달하되, 과도한 감성은 배제

</생각 예시>

---

**[2단계: 메시지 초안 작성]**

위에서 수립한 전략에 따라 아래 규칙을 준수하여 메시지 초안 2개를 작성합니다.

---

### 핵심 혜택 반영 필수
본문에는 **{coreBenefitText}** 안의 모든 내용을  
단 하나도 생략·삭제·변경 없이 자연스럽게 포함해야 합니다.

---

### 메시지 구성 순서
1) **오프닝**  
2) **본문**  
3) **프로모션 기간**  
4) **CTA**

---

### 초안별 규칙

[초안 1: Professional 메시지]

목적: 차분하고 신뢰감 있는 브랜드 톤으로 공식적인 메시지를 전달하는 것

규칙

첫 문장은 확실한 정보 또는 안내 문장으로 시작

과장 표현·감성 표현 금지

톤은 정중·단정·중립적

조건·절차·기준 등 근거 기반 표현 1회 이상 포함

문장은 길지 않게, 명료하게 핵심만 정리

고객 명칭은 ‘고객님’ 또는 중립적 지칭만 사용

---

#### [초안 2: Human 메시지]
목적: 친근하고 사람같은 대화 톤으로 감정적 거리감을 줄이는 것

규칙

첫 문장은 고객 상황 공감 또는 일상적 톤으로 시작

부드러운 표현 2개 이상 사용
(예: “조금 더 편하게”, “가볍게 알려드려요”, “필요하실까 해서”)

딱딱한 공식 표현 금지

문장은 대화하듯 자연스럽게

고객을 직접 지칭하는 2인칭 문체(“고객님”, “지금 필요하실 거예요”) 사용 가능

---

### 두 초안은 반드시 서로 확실히 달라야 합니다.

---

## 메시지는 반드시 아래 형식 그대로 출력해야 합니다.
섹션 제목은 꼭 포함하고, 줄바꿈도 동일하게 유지합니다.

[오프닝]
- 오직 “핵심혜택요약”만 사용하여 1문장
- 고객의 이목을 집중시킬 수 있는 전략을 사용하여 오프닝 문장을 작성하세요  
- 예: "띵동📦 {{고객이름}} 고객님께 {{핵심혜택요약}}이 도착했습니다!"
     -> 택배 문자처럼 보이게 하여 고객이 광고 문자가 아닌 자신에게 필요한 문자처럼 느끼게 하기

[본문]
② **본문 – {coreBenefitText} 기반 전체 재작성**
- {coreBenefitText}의 모든 혜택/내용을 빠짐없이 반영
- 항목이 여러 개면 '-' 로 구분하여 가독성 있게 나열
- 페르소나 특징 기반 설명 1~2문장 포함
         
###  타겟 특성 기반 해석 문단(필수)

아래 변수는 모든 초안에서 반드시 활용해야 합니다:
- **타겟 특징:** {target_features}
- **타겟 특징:** {target_name}

혜택 나열 이후, 반드시 아래 요건을 충족하는  
**“타겟 기반 해석 문단(1~3문장)”**을 추가해야 합니다:

1) {target_features}가 가진 행동·선호·패턴을 직접 언급할 것  
2) 이 타겟이 이번 프로모션에서 **어떤 부분에서 실제 이익을 얻는지** 설명할 것  
3) 이 프로모션이 {target_name} 세그먼트에 **특히 적합한 이유**를 구체적으로 연결지어 기술할 것  
4) 단순 반복 금지 — 반드시 “특징 → 혜택 연결 구조”로 작성
5) {target_name}을 직접 언급하는 것이 아니라 {target_features}의 특징을 이용하기(예: 주말 활동가 -> 주말을 책과 함께 보내시는 고객님)

※ 이 문단이 누락되면 메시지 생성은 실패로 간주합니다.
         

[프로모션 기간]
- {coreBenefitText} 안에서 기간을 직접 추출하여 정확히 작성

[CTA]
👉 자세히 보기: {source_urls}

---


### 초안 작성 규칙 공통
- 두 초안은 반드시 서로 구별되는 톤과 메시지
- {feedback_instructions}

---

## 최종 출력(JSON)
그 어떤 설명도 덧붙이지 말고 아래 형식 그대로 출력하세요:
각 message_text 내부는 반드시 '[오프닝]~[CTA]' 구조를 그대로 포함해야 합니다.

{{
    "drafts": [
        {{
            "message_draft_index": 1,
            "message_text": "(정보성 중심 메시지를 [오프닝]~[CTA] 형식 그대로 작성)"
        }},
        {{
            "message_draft_index": 2,
            "message_text": "(신뢰성 중심 메시지를 [오프닝]~[CTA] 형식 그대로 작성)"
        }}
    ]
}}
""")
])

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
            response = chain.invoke({
                "coreBenefitText": core_benefit_text,
                "source_urls": source_urls_str,
                "feedback_instructions": feedback_instructions,
                "feedback_section": feedback_section,
                "target_name": persona["target_name"],
                "target_features": persona["target_features"],
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
                group_idx = persona["target_group_index"]

                if group_idx in personas_to_rework:
                    all_feedback = "\n".join(feedback_per_persona[group_idx])
                    feedback_instr = "검증 실패 항목을 기준으로 메시지를 재작성하세요."

                    response = chain.invoke({
                        "coreBenefitText": core_benefit_text,
                        "source_urls": source_urls_str,
                        "feedback_instructions": feedback_instr,
                        "feedback_section": all_feedback,
                        "target_name": persona["target_name"],
                        "target_features": persona["target_features"],
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
        response = chain.invoke({
            "coreBenefitText": core_benefit_text,
            "source_urls": source_urls_str,
            "feedback_instructions": "",
            "feedback_section": "",
            "target_name": persona["target_name"],
            "target_features": persona["target_features"],
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
