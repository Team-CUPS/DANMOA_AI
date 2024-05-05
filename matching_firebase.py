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

