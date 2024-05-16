import time
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import AutoModel, AutoTokenizer
import torch
import openai
import logging
import torch


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase and Firestore
cred = credentials.Certificate('danmoa-p5plsh-firebase-adminsdk-kyjdv-f84bfa1051.json')#danmoa-p5plsh-firebase-adminsdk-kyjdv-7d89ae5674.json
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

# Function to calculate similarity between two texts
def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_scores = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
        similarity_score = similarity_scores.mean().item()
    return similarity_score

def update_signal_and_score():
    company_counter = 0  # 추가된 카운터
    while True:
        docs = db.collection('qa').where('signal', '==', 0).get()

        for doc in docs:
            doc_id = doc.id
            print(f"Updating document ID: {doc_id}")
            doc_ref = db.collection('qa').document(doc_id)

            # 입력 형태를 GPT API를 활용해서 모델에 맞게 정제
            # intro_text = doc.get('usr_input_txt')
            
            prompt = doc.get('usr_input_txt') + "\n위에 있는 문장에 대한 답을 한국어로 한문장으로 알려줘."
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            intro_text = response["choices"][0]["message"]["content"]

            # Query relevant job postings
            companies = db.collection('jobs').stream()
            
            scores = []

            for jobs in companies:
                name = jobs.get('COMPANY')
                main_duty = jobs.get('MAINDUTIES')
                qualification = jobs.get('QUALIFICATION')
                preferential = jobs.get('PREFERENTIAL')

                main_duty_score = calculate_similarity(model, tokenizer, intro_text, main_duty)
                qualification_score = calculate_similarity(model, tokenizer, intro_text, qualification)
                preferential_score = calculate_similarity(model, tokenizer, intro_text, preferential)
                total_score = (main_duty_score + qualification_score + preferential_score) / 3

                combined_output = f"Qualification: {qualification}, Preferential: {preferential}"

                scores.append({
                    'score': total_score,
                    'output': combined_output,
                    'name': name
                })

                company_counter += 1  # 카운터 증가
                if company_counter % 100 == 0:
                    print(f"100개 회사 비교 완료, 현재까지 처리한 회사 수: {company_counter}")
                if company_counter >= 1500:  #채용공고수 검색수 조절
                    print("1500개완료")
                    break

            updates = {}
            # Sort scores in descending order and take top 5
            scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:5]

            # Average score
            average_score = sum(score['score'] for score in scores) / len(scores)

            # Top5를 추출해서 합산
            combined_output = "\n\n".join(score['output'] for score in scores)

            cmp_name =  "\n\n".join(score['name'] for score in scores)

            # GPT API로 combined output을 설명하는 것처럼 문장을 정제
            control = "\n위에 있는 전체 문장 중 \"" + intro_text + "\" 이 문장과 관련된 문장을"+ combined_output +"중에 사용해서 공부하도록 조언해주는거처럼 3줄정도로 기술이름이나 필요스택등을 포함해서 한국어로 요약해서 말해줘 단/"+ intro_text +"랑관계있는 문장을 선별해서 3줄로요약해줘"
            prompt = combined_output + control
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
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

        if not docs:
            print("No documents to update. Waiting...")
            time.sleep(10)

# Function call
update_signal_and_score()
