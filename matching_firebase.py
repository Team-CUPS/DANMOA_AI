"""
import time
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import AutoModel, AutoTokenizer
import torch
from nltk.tokenize import sent_tokenize
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase and Firestore
cred = credentials.Certificate('danmoa-p5plsh-firebase-adminsdk-kyjdv-7d89ae5674.json')
firebase_admin.initialize_app(cred, {'projectId': 'danmoa-p5plsh'})
db = firestore.client()

# Initialize the transformer model and tokenizer
model_path = "kazma1/simcse-robertsmall-matching"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Function to calculate similarity between two texts
def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_scores = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
        similarity_score = similarity_scores.mean().item()
    return similarity_score

def update_signal_and_score():
    while True:
        docs = db.collection('ai').where('signal', '==', 0).get()

        for doc in docs:
            doc_id = doc.id
            print(f"Updating document ID: {doc_id}")
            doc_ref = db.collection('ai').document(doc_id)

            intro_text = doc.get('intro_content')
            desire_field = doc.get('desire_field')

            # Query relevant job postings based on 'desire_field'
            job_postings = db.collection('job_postings').where('job_group', '==', desire_field).stream()

            for posting in job_postings:
                job_posting_id = posting.id
                job_posting_content = ' '.join([posting.get('title', ''), posting.get('group_intro', ''),
                                                posting.get('mainduties', ''), posting.get('qualification', ''),
                                                posting.get('preferential', '')])

                mainduties_score = calculate_similarity(model, tokenizer, intro_text, posting.get('mainduties', ''))

                if mainduties_score >= 0.3:
                    print(f"Inserting into MATCH table: {mainduties_score}, {job_posting_id}, {doc_id}")
                    db.collection('match').add({
                        'match_score': mainduties_score,
                        'job_posting_id': job_posting_id,
                        'intro_id': doc_id
                    })
                    continue

            # Update signal after processing
            doc_ref.update({'signal': 2})

        if not docs:
            print("No documents to update. Waiting...")
            time.sleep(10)

# Function call
update_signal_and_score()
"""


"""
#초기버전
import time
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import AutoModel, AutoTokenizer
import torch
from nltk.tokenize import sent_tokenize
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase and Firestore
cred = credentials.Certificate('danmoa-p5plsh-firebase-adminsdk-kyjdv-7d89ae5674.json')
firebase_admin.initialize_app(cred, {'projectId': 'danmoa-p5plsh'})
db = firestore.client()

# Initialize the transformer model and tokenizer
model_path = "kazma1/simcse-robertsmall-matching"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Function to calculate similarity between two texts
def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_scores = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
        similarity_score = similarity_scores.mean().item()
    return similarity_score

def update_signal_and_score():
    while True:
        docs = db.collection('ai').where('signal', '==', 0).get()

        for doc in docs:
            doc_id = doc.id
            print(f"Updating document ID: {doc_id}")
            doc_ref = db.collection('ai').document(doc_id)

            intro_text = doc.get('intro_content')
            desire_field = doc.get('desire_field')

            # Query relevant job postings based on 'desire_field'
            companies = db.collection('company').stream()

            for company in companies:
                name = company.get('name', '')
                main_duty = company.get('main_duty', '')
                qualification = company.get('qualification', '')
                preferential = company.get('preferential', '')

                main_duty_score = calculate_similarity(model, tokenizer, intro_text, main_duty)
                qualification_score = calculate_similarity(model, tokenizer, intro_text, qualification)
                preferential_score = calculate_similarity(model, tokenizer, intro_text, preferential)
                total_score = (main_duty_score + qualification_score + preferential_score) / 3

                combined_output = f"Main Duty: {main_duty}, Qualification: {qualification}, Preferential: {preferential}"

                # Update the document with the calculated score and combined output
                doc_ref.update({
                    'signal': 2,
                    'ai_score': total_score,
                    'ai_output': combined_output
                })
                break  # Break after finding the first matching company (you can adjust this logic as needed)

        if not docs:
            print("No documents to update. Waiting...")
            time.sleep(10)

# Function call
update_signal_and_score()

"""
"""
import time
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import AutoModel, AutoTokenizer
import torch
from nltk.tokenize import sent_tokenize
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase and Firestore
cred = credentials.Certificate('danmoa-p5plsh-firebase-adminsdk-kyjdv-7d89ae5674.json')
firebase_admin.initialize_app(cred, {'projectId': 'danmoa-p5plsh'})
db = firestore.client()

# Initialize the transformer model and tokenizer
model_path = "kazma1/simcse-robertsmall-matching"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Function to calculate similarity between two texts
def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_scores = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
        similarity_score = similarity_scores.mean().item()
    return similarity_score

def update_signal_and_score():
    while True:
        docs = db.collection('qa').where('signal', '==', 0).get()

        for doc in docs:
            doc_id = doc.id
            print(f"Updating document ID: {doc_id}")
            doc_ref = db.collection('qa').document(doc_id)

            intro_text = doc.get('usr_input_txt')

            # Query relevant job postings
            companies = db.collection('jobs').stream()

            scores = []

            for jobs in companies:
                name = jobs.get('COMPANY', '')
                main_duty = jobs.get('MAINDUTIES', '')
                qualification = jobs.get('QUALIFICATION', '')
                preferential = jobs.get('PREFERENTIAL', '')

                main_duty_score = calculate_similarity(model, tokenizer, intro_text, main_duty)
                qualification_score = calculate_similarity(model, tokenizer, intro_text, qualification)
                preferential_score = calculate_similarity(model, tokenizer, intro_text, preferential)
                total_score = (main_duty_score + qualification_score + preferential_score) / 3

                combined_output = f"Main Duty: {main_duty}, Qualification: {qualification}, Preferential: {preferential}"

                scores.append({
                    'score': total_score,
                    'output': combined_output
                })

            updates ={}
            # Sort scores in descending order and take top 5
            scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:5]

            #average score
            average_score = sum(score['score'] for score in scores) / len(scores)
            # Update the document with the top 5 scores and outputs
       
            #for i, score_info in enumerate(scores):
               # updates[f'ai_score{i+1}'] = score_info['score']
               
            #updates['ai_output'] = f"{updates.get('ai_output', '')}\n\n{score_info['output']}" if updates.get('ai_output') else score_info['output']
            combined_output = "\n\n".join(score['output'] for score in scores)
            updates = {
                'ai_score': average_score,
                'ai_output': combined_output,
                'signal': 2
            }
        if not docs:
            print("No documents to update. Waiting...")
            time.sleep(10)

# Function call
update_signal_and_score()
import time
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import AutoModel, AutoTokenizer
import torch
import openai
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase and Firestore
cred = credentials.Certificate('danmoa-p5plsh-firebase-adminsdk-kyjdv-7d89ae5674.json')
firebase_admin.initialize_app(cred, {'projectId': 'danmoa-p5plsh'})
db = firestore.client()

# Initialize OpenAI API
openai.api_key = "your_openai_api_key"

# Initialize the transformer model and tokenizer
model_path = "kazma1/simcse-robertsmall-matching"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Function to calculate similarity between two texts
def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_scores = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
        similarity_score = similarity_scores.mean().item()
    return similarity_score

def update_signal_and_score():
    while True:
        docs = db.collection('qa').where('signal', '==', 0).get()

        for doc in docs:
            doc_id = doc.id
            print(f"Updating document ID: {doc_id}")
            doc_ref = db.collection('qa').document(doc_id)

            intro_text = doc.get('usr_input_txt')

            # Query relevant job postings
            companies = db.collection('jobs').stream()

            scores = []

            for jobs in companies:
                name = jobs.get('COMPANY', '')
                main_duty = jobs.get('MAINDUTIES', '')
                qualification = jobs.get('QUALIFICATION', '')
                preferential = jobs.get('PREFERENTIAL', '')

                main_duty_score = calculate_similarity(model, tokenizer, intro_text, main_duty)
                qualification_score = calculate_similarity(model, tokenizer, intro_text, qualification)
                preferential_score = calculate_similarity(model, tokenizer, intro_text, preferential)
                total_score = (main_duty_score + qualification_score + preferential_score) / 3

                combined_output = f"Main Duty: {main_duty}, Qualification: {qualification}, Preferential: {preferential}"

                scores.append({
                    'score': total_score,
                    'output': combined_output
                })

            updates = {}
            # Sort scores in descending order and take top 5
            scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:5]

            # Average score
            average_score = sum(score['score'] for score in scores) / len(scores)

            # Combine the outputs of the top 5 scores
            combined_output = "\n\n".join(score['output'] for score in scores)

            # Summarize the combined output using OpenAI
            prompt = f"Combine the following job requirements into a brief summary:\n\n{combined_output}"
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
"""
import time
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import AutoModel, AutoTokenizer
import torch
import openai
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase and Firestore
cred = credentials.Certificate('danmoa-p5plsh-firebase-adminsdk-kyjdv-7d89ae5674.json')
firebase_admin.initialize_app(cred, {'projectId': 'danmoa-p5plsh'})
db = firestore.client()

# Initialize OpenAI API
openai.organization = ""
openai.api_key = ""

# Initialize the transformer model and tokenizer
model_path = "kazma1/simcse-robertsmall-matching"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Function to calculate similarity between two texts
def calculate_similarity(model, tokenizer, text1, text2):
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity_scores = torch.nn.functional.cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[1], dim=1)
        similarity_score = similarity_scores.mean().item()
    return similarity_score

def update_signal_and_score():
    while True:
        docs = db.collection('qa').where('signal', '==', 0).get()

        for doc in docs:
            doc_id = doc.id
            print(f"Updating document ID: {doc_id}")
            doc_ref = db.collection('qa').document(doc_id)

            intro_text = doc.get('usr_input_txt')

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

            updates = {}
            # Sort scores in descending order and take top 5
            scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:5]

            # Average score
            average_score = sum(score['score'] for score in scores) / len(scores)

            # Combine the outputs of the top 5 scores
            combined_output = "\n\n".join(score['output'] for score in scores)

            cmp_name =  "\n\n".join(score['name'] for score in scores)

            # Summarize the combined output using OpenAI
            prompt = combined_output +"\n위에 있는 문장을 참고해서 공부하도록 조언해주는거처럼 한국어로 요약해서 말해줘"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_output = response["choices"][0]["message"]["content"]

            # Update the document with the average score and the summarized output
            updates = {
                'ai_score': average_score,
                'ai_output': cmp_name+"\n"+ai_output,
                'signal': 2
            }

            doc_ref.update(updates)

        if not docs:
            print("No documents to update. Waiting...")
            time.sleep(10)

# Function call
update_signal_and_score()
