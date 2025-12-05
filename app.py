import json
import math
from datetime import datetime
from flask import Flask, request, jsonify
from agent_workflow import build_agent_workflow, CampaignState
from typing import Dict, Any
from rag_utils import add_document_to_chroma, query_chroma, get_all_documents_from_chroma, update_document_in_chroma, delete_document_from_chroma, get_or_create_collection, get_document_by_id
from rag_utils import collection 

app = Flask(__name__)

# LangGraph 워크플로우를 서버 시작 시 한 번 로드합니다.
# 실제 운영 환경에서는 이 부분을 더 견고하게 처리할 수 있습니다.
try:
    workflow_app = build_agent_workflow()
    print("LangGraph workflow loaded successfully.")
except Exception as e:
    print(f"Error loading LangGraph workflow: {e}")
    workflow_app = None

@app.route('/api/generate', methods=['POST'])
def run_workflow():
    if workflow_app is None:
        return jsonify({"error": "LangGraph workflow is not loaded."}), 500

    data: Dict[str, Any] = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON input."}), 400

    # CampaignState의 input_data 필드에 해당하는 데이터를 추출합니다.
    # 클라이언트로부터 받은 JSON 데이터가 곧 input_data가 됩니다.
    initial_input_data = data

    # LangGraph 워크플로우의 초기 상태를 설정합니다.
    # rework_count 등은 워크플로우 내부에서 관리되므로 초기에는 기본값으로 설정합니다.
    initial_state: CampaignState = {
        "input_data": initial_input_data,
        "target_personas": None,
        "messages_drafts": None,
        "validation_reports": None,
        "rework_count": 0,
        "validator_feedback": None,
        "refine_feedback": None # refine을 위한 필드
    }

    try:
        # 워크플로우를 실행하고 최종 상태를 가져옵니다.
        # stream() 대신 invoke()를 사용하여 최종 결과만 가져옵니다.
        final_state = workflow_app.invoke(initial_state)
        
        print(f"DEBUG: Type of final_state: {type(final_state)}")
        print(f"DEBUG: Content of final_state: {final_state}")
        # final_state에서 'final_output' 키를 찾아 반환합니다.
        # run_formatter_agent에서 이 키로 최종 결과를 저장했습니다.
        if 'final_output' in final_state:
            return jsonify({"target_groups": final_state['final_output']}), 200
        else:
            return jsonify({"error": "Workflow completed but final_output not found in state."}), 500

    except Exception as e:
        print(f"Error running workflow: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/campaigns/<string:campaign_id>/refine', methods=['POST'])
def refine_workflow(campaign_id):
    if workflow_app is None:
        return jsonify({"error": "LangGraph workflow is not loaded."}), 500

    data: Dict[str, Any] = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON input."}), 400

    # BE에서 받은 campaign_context와 feedback_text를 추출합니다.
    campaign_context = data.get("campaign_context")
    feedback_text = data.get("feedback_text")
    # target_personas도 받아서, 있으면 targeting 단계를 건너뛸 수 있도록 합니다.
    target_personas = data.get("target_personas")

    if not campaign_context:
        return jsonify({"error": "Missing 'campaign_context' in request body. Cannot refine without original campaign data."}), 400
    if not feedback_text:
        return jsonify({"error": "Missing 'feedback_text' in request body. Please provide feedback for refinement."}), 400

    # refine_feedback을 LangGraph 상태에 맞는 형식으로 구성합니다.
    refine_feedback_details = {"details": feedback_text} if feedback_text else None

    # LangGraph 워크플로우의 초기 상태를 설정합니다.
    # target_personas가 있으면 targeting 단계를 건너뛰고, 없으면 처음부터 다시 실행합니다.
    print(f"DEBUG: Received target_personas for refine: {'Yes' if target_personas else 'No'}")
    initial_state: CampaignState = {
        "input_data": campaign_context,
        "target_personas": target_personas,  # 페르소나를 전달하여 targeting 건너뛰기
        "messages_drafts": None, # messaging을 다시 거치므로 None으로 설정
        "refine_feedback": refine_feedback_details, # 마케터의 수정 요청 피드백
        "validation_reports": None,
        "rework_count": 0,
        "validator_feedback": None,
    }

    try:
        # 워크플로우를 'targeting' 노드부터 시작하여 실행합니다.
        # 이렇게 하면 마케터의 피드백이 페르소나 생성 단계부터 반영됩니다.
        final_state = workflow_app.invoke(initial_state)
        
        print(f"DEBUG: Type of final_state after refine: {type(final_state)}")
        print(f"DEBUG: Content of final_state after refine: {final_state}")

        if 'final_output' in final_state and final_state['final_output']:
            return jsonify({"target_groups": final_state['final_output']}), 200
        else:
            return jsonify({"error": "Workflow (refine) completed but final_output not found."}), 500

    except Exception as e:
        print(f"Error running refine workflow: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge', methods=['POST'])
def handle_knowledge():
    """
    ChromaDB에 지식을 추가하거나 검색합니다.
    요청 본문에 'campaign_summary'가 있으면 지식 추가로,
    'query_texts'가 있으면 지식 검색으로 동작합니다.
    """
    data = request.get_json()
    print(f"DEBUG: Received data for /api/knowledge: {data}")
    if not data:
        return jsonify({"error": "Invalid JSON input."}), 400

    # 지식 추가 (새로운 포맷 - content_text 기반)
    if 'content_text' in data and 'related_campaign_id' in data:
        try:
            campaign_id = data['related_campaign_id']
            doc_to_add = data['content_text']
            
            # 메타데이터 구성
            metadata_to_add = {
                "campaign_id": campaign_id,
                "status": "successful"
            }
            if 'title' in data:
                metadata_to_add['title'] = data['title']
            if 'source_type' in data:
                metadata_to_add['source_type'] = data['source_type']
            if 'registration_date' in data: # registration_date 추가
                metadata_to_add['registration_date'] = data['registration_date']

            id_to_add = f"campaign-{campaign_id}"

            add_document_to_chroma(
                documents=[doc_to_add],
                metadatas=[metadata_to_add],
                ids=[id_to_add]
            )
            return jsonify({"message": f"Success case for campaign {campaign_id} added successfully (new format)."}), 201
        except KeyError as e:
            print(f"DEBUG: KeyError in /api/knowledge (new format): {e}")
            return jsonify({"error": f"Missing required field for adding knowledge (new format): {e}"}), 400
        except Exception as e:
            print(f"Error adding success case to Chroma DB (new format): {e}")
            return jsonify({"error": str(e)}), 500

    # 지식 추가 (기존 포맷 - campaign_summary 기반)
    elif 'campaign_summary' in data:
        try:
            campaign_id = data['campaign_id']
            summary = data['campaign_summary']
            details = data.get('campaign_details', {})

            # ChromaDB는 메타데이터 값으로 리스트를 지원하지 않으므로, 문자열로 변환합니다.
            if 'custom_columns' in details and isinstance(details['custom_columns'], list):
                details['custom_columns'] = ', '.join(details['custom_columns'])

            doc_to_add = summary
            metadata_to_add = {
                "campaign_id": campaign_id,
                "status": "successful",
                **details
            }
            if 'registration_date' in data: # registration_date 추가
                metadata_to_add['registration_date'] = data['registration_date']
            id_to_add = f"campaign-{campaign_id}"

            add_document_to_chroma(
                documents=[doc_to_add],
                metadatas=[metadata_to_add],
                ids=[id_to_add]
            )
            return jsonify({"message": f"Success case for campaign {campaign_id} added successfully."}), 201
        except KeyError as e:
            print(f"DEBUG: KeyError in /api/knowledge: {e}")
            return jsonify({"error": f"Missing required field for adding knowledge: {e}"}), 400
        except Exception as e:
            print(f"Error adding success case to Chroma DB: {e}")
            return jsonify({"error": str(e)}), 500

    # 지식 검색
    elif 'query_texts' in data:
        try:
            n_results = data.get('n_results', 5)
            where_filter = data.get('where_filter')
            
            results = query_chroma(
                query_texts=data['query_texts'],
                n_results=n_results,
                where_filter=where_filter
            )
            return jsonify({"results": results}), 200
        except Exception as e:
            print(f"Error querying Chroma DB: {e}")
            return jsonify({"error": str(e)}), 500
            
    # 잘못된 요청
    else:
        return jsonify({"error": "Invalid request body. For adding knowledge, provide 'campaign_summary' or 'content_text'."}), 400

@app.route('/api/knowledge', methods=['GET'])
def search_or_list_knowledge():
    """
    URL 쿼리 파라미터를 사용하여 ChromaDB의 지식을 검색하거나 조회합니다.
    - 'q': 의미 검색(semantic search)
    - '[key]__contains=[value]': 메타데이터 'key'에 'value'가 포함된 문서 검색
    - '[key]=[value]': 메타데이터 'key'가 'value'와 정확히 일치하는 문서 검색

    Query Parameters:
        - q (str, optional): 의미 검색을 위한 쿼리 텍스트.
        - n_results (int, optional): 'q' 사용 시 반환할 최대 결과 수 (기본값: 5).
        - page (int, optional): 'q'가 없을 때 조회할 페이지 번호 (기본값: 1).
        - size (int, optional): 'q'가 없을 때 한 페이지에 포함할 문서 수 (기본값: 10).
    """
    try:
        args = request.args
        search_query = args.get('q')

        # 1. 필터 조건 분리: exact_match vs contains
        reserved_keys = ['q', 'n_results', 'page', 'size']
        exact_filter_conditions = []
        contains_filters = {}
        
        for key, value in args.items():
            if not key or not value:
                continue

            if key in reserved_keys:
                continue
            
            if key.endswith('__contains'):
                field_name = key.removesuffix('__contains')
                contains_filters[field_name] = value
            else:
                exact_filter_conditions.append({key: value})

        # 2. ChromaDB용 where_filter 구성 (Exact-match 전용)
        where_filter = None
        if len(exact_filter_conditions) == 1:
            where_filter = exact_filter_conditions[0]
        elif len(exact_filter_conditions) > 1:
            where_filter = {"$and": exact_filter_conditions}

        # 3. 'q' 파라미터가 있는 경우: 의미 검색 후 후처리 필터링
        if search_query:
            n_results = args.get('n_results', 5, type=int)
            
            # 의미 검색 실행 (정확 일치 필터만 적용)
            # contains 필터를 위해 더 많은 결과를 요청할 수 있습니다. (예: n_results * 5)
            results = query_chroma(
                query_texts=[search_query],
                n_results=n_results * 5, 
                where_filter=where_filter
            )

            # 후처리: 'contains' 필터 적용
            if contains_filters:
                final_results = []
                for doc in results:
                    all_match = True
                    for key, value in contains_filters.items():
                        if value.lower() not in doc.get('metadata', {}).get(key, '').lower():
                            all_match = False
                            break
                    if all_match:
                        final_results.append(doc)
                results = final_results[:n_results]

            return jsonify({"results": results}), 200

        # 4. 'q' 파라미터가 없는 경우: 전체 목록 조회 후 후처리 필터링
        else:
            # DB에서 정확 일치 필터로 문서 조회
            all_documents = get_all_documents_from_chroma(where_filter=where_filter)
            
            # 후처리: 'contains' 필터 적용
            if contains_filters:
                filtered_documents = []
                for doc in all_documents:
                    all_match = True
                    for key, value in contains_filters.items():
                         if value.lower() not in doc.get('metadata', {}).get(key, '').lower():
                            all_match = False
                            break
                    if all_match:
                        filtered_documents.append(doc)
                all_documents = filtered_documents

            # 페이지네이션 적용
            page = args.get('page', 1, type=int)
            size = args.get('size', 10, type=int)
            if page < 1: page = 1
            
            total_documents = len(all_documents)
            total_pages = math.ceil(total_documents / size) if size > 0 else 0

            start_index = (page - 1) * size
            end_index = start_index + size
            paginated_docs = all_documents[start_index:end_index]

            response = {
                "total_documents": total_documents,
                "total_pages": total_pages,
                "current_page": page,
                "page_size": size,
                "filters": {"exact": where_filter, "contains": contains_filters},
                "data": paginated_docs
            }
            return jsonify(response), 200

    except Exception as e:
        print(f"Error in GET /api/knowledge: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/<string:doc_id>', methods=['GET'])
def get_knowledge_by_id(doc_id):
    """
    ChromaDB에 저장된 특정 지식 문서를 ID로 조회합니다.
    """
    try:
        document = get_document_by_id(doc_id)
        if document:
            return jsonify(document), 200
        else:
            return jsonify({"error": f"Document with ID '{doc_id}' not found."}), 404
    except Exception as e:
        print(f"Error retrieving document from Chroma DB by ID: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/<string:doc_id>', methods=['PUT'])
def update_knowledge(doc_id):
    """
    ChromaDB에 저장된 특정 지식 문서를 업데이트합니다.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON input."}), 400

    document = data.get('document')
    metadata = data.get('metadata')

    if not document or not metadata:
        return jsonify({"error": "Missing 'document' or 'metadata' in request body."}), 400

    # updated_at 필드를 현재 시간으로 추가 또는 업데이트
    metadata['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        update_document_in_chroma(doc_id, document, metadata)
        return jsonify({"message": f"Document with ID '{doc_id}' updated successfully."}), 200
    except Exception as e:
        print(f"Error updating document in Chroma DB: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/<string:doc_id>', methods=['DELETE'])
def delete_knowledge(doc_id):
    """
    ChromaDB에 저장된 특정 지식 문서를 삭제합니다.
    """
    try:
        delete_document_from_chroma(doc_id)
        return jsonify({"message": f"Document with ID '{doc_id}' deleted successfully."}), 200
    except Exception as e:
        print(f"Error deleting document from Chroma DB: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/knowledge/count', methods=['GET'])
def get_knowledge_count():
    try:
        # ChromaDB 컬렉션의 count() 메서드를 사용하여 전체 문서 수를 가져옵니다.
        # collection 변수는 rag_utils.py에서 get_or_create_collection을 통해 초기화됩니다.
        collection_count = collection.count()
        
        # JSON 형식으로 개수를 반환합니다.
        return jsonify({"count": collection_count}), 200
        
    except Exception as e:
        print(f"Error counting documents in ChromaDB: {e}")
        return jsonify({"error": "Failed to count documents"}), 500

@app.route('/')
def health_check():
    return "AI Agent Flask Server is running!", 200

if __name__ == '__main__':
    # Flask 서버를 실행합니다.
    # 개발 환경에서는 debug=True로 설정할 수 있습니다.
    # 실제 운영 환경에서는 Gunicorn과 같은 WSGI 서버를 사용하는 것이 좋습니다.
    app.run(host='0.0.0.0', port=5000, debug=True)
