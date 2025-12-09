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
    validator_feedback: Union[Dict, None] # Validator가 Messaging Agent에게 전달할 구체적인 수정 피드백
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
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
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
    """
    Messaging Agent: 각 타겟 페르소나에 맞춰 초개인화 메시지 초안 2개를 생성합니다.
    Validator Agent로부터의 피드백을 반영하여 메시지를 재생성할 수 있습니다.
    """
    print("---" + " Messaging Agent 실행 중 ---")
    input_data = state.get('input_data', {})
    target_personas = state.get('target_personas', [])
    rework_count = state.get('rework_count', 0)
    validator_feedback = state.get('validator_feedback', None)
    refine_feedback = state.get('refine_feedback', None)  # 마케터 피드백 추가
    core_benefit_text = input_data.get('coreBenefitText', '기본 혜택')  # 'coreBenefitText'로 수정
    custom_columns_data = input_data.get('customColumns', {})  # 'customColumns'로 수정하고 객체로 받음
    source_urls = input_data.get('sourceUrls', []) # sourceUrls 추가
    source_urls_str = ", ".join(source_urls) if source_urls else '없음'

    # 새로운 customColumns 형식(객체)을 LLM이 이해하기 좋은 문자열로 변환
    if isinstance(custom_columns_data, dict):
        columns_list = []
        for key, value in custom_columns_data.items():
            # 키는 그대로 사용하고, 값은 예시 또는 타입으로 설명
            columns_list.append(f"- `{{{key}}}`: ({value})")
        columns_for_prompt = "\n".join(columns_list)
    else:
        # 기존 형식(리스트) 또는 기본값 처리
        columns_for_prompt = ", ".join(custom_columns_data)


    # LLM 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        (
        "system",
        """
        당신은 고객의 감정을 움직이는 초개인화 마케팅 메시지 전문 카피라이터입니다.
        아래 규칙에 따라 타겟 페르소나에게 맞춘 메시지 초안을 2개 생성합니다.

        ----------------------------------------------------------------
        [1] 메시지 전체 구조
        ----------------------------------------------------------------

        ① **오프닝 문장 (띵동 문구)**
        - 오직 “핵심혜택요약”만 사용하여 1문장으로 표현합니다.
        - 예: "띵동📦 {{고객이름}} 고객님께 {{핵심혜택요약}}이 도착했습니다!"

        ※ 오프닝에는 전체 프로모션 내용을 넣지 마십시오.

        ---------------------------------------------------------------

        ② **본문 – 제공된 프로모션 상세 내용 전체를 기반으로 재작성 (가장 중요)**

        ※ 매우 중요: 본문 작성 시 마케터가 제공한 프로모션 상세 내용 안의  
        모든 항목(혜택, 쿠폰, 구독 서비스, 특징, 전략, 부가 내용 포함)을  
        **단 하나도 생략하지 말고 반드시 모두 반영하여** 자연스러운 마케팅 문체로 재구성해야 합니다.

        본문 구성 규칙:
        1) “프로모션 전체 개요”를 1~2문장으로 요약
        2) 상세 구성 요소를 항목별로 자연스럽게 모두 포함  
        (스트리밍, 전자책, 쿠폰, 포인트 등 혜택 목록 모두 포함)
        3) 타겟 페르소나의 특징을 반영하여  
        “왜 이 프로모션이 이 고객에게 유리한지”를 2~3문장으로 설명
        4) 시즌성·전환율 전략 등 마케터가 입력한 배경 및 의도도 자연스럽게 포함

        ---------------------------------------------------------------

        ③ [프로모션 기간] 
        - 반드시 입력된 프로모션 상세 내용 안에서 기간 정보를 직접 분석하여 포함해야 합니다.
        - 별도의 변수로 기간이 전달되지 않으므로, 제공된 프로모션 텍스트에서 기간을 찾아 문장으로 작성하십시오.
        - 예: “📅 이벤트 기간: 2025.12.20 ~ 12.31”

        ---------------------------------------------------------------

        ④ **URL 제공 시 CTA**
        - 제공된 URL(source_urls)이 '없음'이 아니라면 아래 형식으로 출력:
        예: “👉 자세히 보기: {source_urls}”

        ---------------------------------------------------------------

        ⑤ **커스텀 변수(이름 등) 활용**
        - 이름 등 최소 1개 컬럼을 본문에 자연스럽게 사용합니다.

        ---------------------------------------------------------------

        [2] 초안 작성 규칙
        ---------------------------------------------------------------
        - 두 개의 메시지 초안은 서로 다른 톤/구조로 작성합니다.
        - RAG 성공 사례가 존재하면 톤/전략/표현을 참고해 반영합니다.
        - {feedback_instructions}

        ---------------------------------------------------------------
        [3] 출력(JSON)
        ---------------------------------------------------------------
        아래 형식을 따라야 하며 JSON 객체 하나만 반환해야 합니다.

        {{
        "drafts": [
        {{
            "message_draft_index": 1,
            "message_text": "(전체 메시지 텍스트)"
        }},
        {{
            "message_draft_index": 2,
            "message_text": "(전체 메시지 텍스트)"
        }}
        ]
        }}

        위의 모든 규칙을 철저히 준수해주세요.

        [출력 포맷 규칙 - 매우 중요]

        메시지는 반드시 다음 시각적 구조를 따라 작성해야 합니다.

        ① 오프닝 (띵동 문장)
        - 한 문장, 임팩트 있는 형태

        ② 간단한 인사 + 프로모션 소개 문단
        - 1~2문장

        ③ [제공 혜택] 섹션
        - 반드시 리스트 형태로 정리한다
        - 각 혜택은 다음 구조 사용:
        - "- 혜택명"
            "→ 혜택에 대한 1줄 설명"

        예시:
        - 프리미엄 스트리밍 1개월  
        → 최신 영화/드라마 고화질 무제한 시청  
        - 전자책 구독 1개월  
        → 약 20만 종 콘텐츠 제공  

        ④ [이런 고객님께 추천 섹션]
        - 이 섹션은 반드시 제공된 타겟 페르소나 특징을 기반으로 작성해야 합니다.
        - 프로모션 내용을 보고 일반적인 타겟을 추론하여 작성하는 것을 금지합니다.
        - 타겟 페르소나 특징을 최소 두 문장 이상 직접적으로 언급하며,
        해당 프로모션이 왜 이 타겟에게 특히 잘 맞는지 명확히 연결해 설명해야 합니다.
        - 타겟 페르소나의 속성(예: 관심사, 소비 패턴, 행동, 연령대, 사용 습관 등)을 그대로 활용합니다.

        ⑤ [프로모션 기간]
        - 별도 줄, 별도 섹션 형태

        ⑥ [URL 또는 CTA 버튼]
        - 마지막 줄에 배치

        전체 메시지는 ‘한 문단으로 뭉치지 않고’  
        **반드시 각 섹션 사이에 줄바꿈(빈 줄 1칸 이상)을 포함해야 합니다.**

        출력 예시는 아래와 같은 레이아웃을 준수합니다:
        - 오프닝 → 빈 줄
        - 소개 → 빈 줄
        - 혜택 리스트 → 빈 줄
        - 타겟 연결 → 빈 줄
        - 기간 → 빈 줄
        - URL

        """)
    ])

    # LangChain Expression Language (LCEL) 체인 구성
    chain = prompt | llm | json_parser

    messages_drafts = []
    for persona in target_personas:
        target_name = persona['target_name']
        target_features = persona['target_features']

        # RAG Tool 호출: 성공 사례 검색
        success_case_knowledge = rag_search(query=f"{target_name} 타겟 메시지 성공 사례", source_type='성공 사례')
        print(f"Messaging Agent - RAG Knowledge for {target_name}: {success_case_knowledge}")

        # 피드백 처리
        feedback_instructions = ""
        feedback_section = ""
        if refine_feedback:  # 마케터의 피드백을 최우선으로 반영
            print(f"Messaging Agent - Marketer's Refine Feedback 반영 중: {refine_feedback}")
            feedback_instructions = "이전 초안에 대한 아래의 마케터 피드백을 반영하여 메시지를 수정해주세요."
            feedback_section = f"마케터 수정 피드백: {refine_feedback.get('details', '없음')}"
        elif validator_feedback:  # 마케터 피드백이 없을 경우, Validator 피드백 반영
            print(f"Messaging Agent - Validator Feedback 반영 중: {validator_feedback}")
            feedback_instructions = "이전 초안에 대한 아래의 피드백을 반영하여 메시지를 수정해주세요."
            feedback_section = f"수정 피드백: {validator_feedback.get('details', '없음')}"
            

        # 체인 실행
        response = chain.invoke({
            "feedback_instructions": feedback_instructions,
            "target_name": target_name,
            "target_features": target_features,
            "core_benefit": core_benefit_text,
            "promotion_details": input_data.get("promotionDetails", "N/A"),
            "columns": columns_for_prompt,
            "source_urls": source_urls_str,
            "rag_knowledge": success_case_knowledge,
            "feedback_section": feedback_section
        })

        # 생성된 초안을 persona 정보와 함께 저장
        messages_drafts.append({
            "target_group_index": persona['target_group_index'],
            "target_name": target_name,
            "message_drafts": response.get("drafts", [])
        })
        
    print(f"Messaging Agent - 생성된 메시지 초안: {messages_drafts}")
    # validator_feedback이 있었다면 rework_count를 1 증가시킴
    return {"messages_drafts": messages_drafts, "rework_count": rework_count + 1 if validator_feedback else rework_count}

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
        {{
            "spam_risk_score": <0-100 사이의 정수>,
            "policy_compliance": "<'PASS' 또는 'FAIL'>",
            "review_summary": "<검토 요약>",
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
    needs_rework = False
    validator_feedback = {"reason": "초안 메시지 검증 결과, 수정이 필요합니다.", "details": []}

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

            # 검증 실패 조건 확인
            if report.get('policy_compliance') == 'FAIL' or report.get('spam_risk_score', 0) > 70:
                needs_rework = True
                feedback_detail = (
                    f"타겟 '{target_name}'의 메시지 초안 {draft['message_draft_index']}: "
                    f"{report.get('recommended_action', '피드백 없음')}"
                )
                validator_feedback['details'].append(feedback_detail)

            # 전체 리포트 저장
            report['target_group_index'] = target_group_drafts['target_group_index']
            report['message_draft_index'] = draft['message_draft_index']
            validation_reports.append(report)

    print(f"Validator Agent - 생성된 검증 리포트: {validation_reports}")

    if needs_rework:
        # 피드백의 details를 하나의 문자열로 합침
        feedback_str = "\n".join(validator_feedback['details'])
        return {"validation_reports": validation_reports, "validator_feedback": {"details": feedback_str}}
    else:
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
    validator_feedback = state.get('validator_feedback', None)

    # 최대 재시도 횟수 (1회) 초과 시 강제 종료
    if rework_count >= 1:
        print(f"재시도 횟수 {rework_count}회 초과. Formatter로 이동하여 강제 종료.")
        return "formatter"

    # 검증 실패 조건 확인 (예: policy_compliance == 'FAIL' 또는 스팸 점수 기준 초과)
    # 하나라도 FAIL이거나 스팸 점수가 높으면 재작업 필요
    needs_rework = False
    if validator_feedback and validator_feedback.get('details'):
        needs_rework = True

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
            "core_benefit_text": "KT 5G 프리미엄 요금제, 데이터 완전 무제한!",
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
