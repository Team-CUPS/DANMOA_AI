import time
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import AutoModel, AutoTokenizer
import torch
import openai
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase and Firestore
cred = credentials.Certificate('danmoa-p5plsh-firebase-adminsdk-kyjdv-f84bfa1051.json')  # Firebase 인증서 경로
firebase_admin.initialize_app(cred, {'projectId': 'danmoa-p5plsh'})
db = firestore.client()

# Initialize OpenAI API
openai.organization = ""
openai.api_key = ""

# Initialize the transformer model and tokenizer
model_path = "kazma1/simcse-robertsmall-matching"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Check if the model is on GPU
if next(model.parameters()).is_cuda:
    print("Model is using GPU")
else:
    print("Model is using CPU")

# Function to calculate similarity between two texts
def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU if available
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_scores = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
        similarity_score = similarity_scores.mean().item()
    return similarity_score

def process_company(company, intro_text, model, tokenizer, device):
    name = company.get('COMPANY')
    main_duty = company.get('MAINDUTIES')
    qualification = company.get('QUALIFICATION')
    preferential = company.get('PREFERENTIAL')

    main_duty_score = calculate_similarity(model, tokenizer, intro_text, main_duty)
    qualification_score = calculate_similarity(model, tokenizer, intro_text, qualification)
    preferential_score = calculate_similarity(model, tokenizer, intro_text, preferential)
    total_score = (main_duty_score + qualification_score + preferential_score) / 3

    combined_output = f"Qualification: {qualification}, Preferential: {preferential}"

    return {
        'score': total_score,
        'output': combined_output,
        'name': name
    }

def update_signal_and_score():
    while True:
        docs = db.collection('qa').where('signal', '==', 0).get()

        for doc in docs:
            company_counter = 0  # 각 문서 처리 시작 시 카운터 초기화
            doc_id = doc.id
            print(f"Updating document ID: {doc_id}")
            doc_ref = db.collection('qa').document(doc_id)

            # 입력 형태를 GPT API를 활용해서 모델에 맞게 정제
            prompt = doc.get('usr_input_txt') + "\n위에 있는 문장에 대한 답을 한국어로 한문장으로 알려줘."
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            intro_text = response["choices"][0]["message"]["content"]

            # Query relevant job postings
            companies = list(db.collection('jobs').stream())
            
            scores = []

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_company, company, intro_text, model, tokenizer, device) for company in companies]
                for future in as_completed(futures):
                    result = future.result()
                    scores.append(result)

                    company_counter += 1  # 카운터 증가
                    if company_counter % 100 == 0:
                        print(f"100개 회사 비교 완료, 현재까지 처리한 회사 수: {company_counter}")
                    if company_counter >= 1000:  # 채용공고수 검색수 조절
                        print("1000개완료")
                        break

            if company_counter >= 1000:
                # Sort scores in descending order and take top 5
                scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:5]

                # Average score
                average_score = sum(score['score'] for score in scores) / len(scores)

                # Top5를 추출해서 합산
                combined_output = "\n\n".join(score['output'] for score in scores)
                cmp_name = "\n\n".join(score['name'] for score in scores)

                # GPT API로 combined output을 설명하는 것처럼 문장을 정제
                if average_score <= 0.36:
                    ai_output = "좀 더 정확히 써주시면 자세히 알려드리겠습니다."
                else:
                    control = "\n위에 있는 전체 문장 중 \n" + intro_text + "\n 이 문장과 관련된 내용으로 기술이름이나 필요스택등을 포함해서 한국어로 3~5줄정도 요약한걸 남에게 조언하는 것처럼 말해줘."
                    prompt = combined_output + control
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ai_output = response["choices"][0]["message"]["content"]

                # Update the document with the average score and the summarized output
                updates = {
                    'ai_score': average_score,
                    'ai_output': ai_output,
                    'signal': 2
                }

                doc_ref.update(updates)
                print(f"Document ID: {doc_id} has been updated with average score: {average_score}")
                break

        if not docs:
            print("No documents to update. Waiting...")
            time.sleep(10)

# Function call
update_signal_and_score()
