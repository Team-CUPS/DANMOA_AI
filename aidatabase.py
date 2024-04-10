import time
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials


# Firestore DB 초기화는 이미 완료되었다고 가정합니다.
# Use the application default credentials
cred = credentials.Certificate('danmoa-p5plsh-firebase-adminsdk-kyjdv-7d89ae5674.json')
firebase_admin.initialize_app(cred, {
  'projectId': 'danmoa-p5plsh',
})

db = firestore.client()

def update_signal_and_score():
    # 무한 루프를 돌면서 작업을 수행합니다.
    while True:
        # ai 컬렉션에서 signal이 0인 문서를 쿼리합니다.
        docs = db.collection('ai').where('signal', '==', 0).get()

        for doc in docs:
            print(f"Updating document ID: {doc.id}")
            doc_ref = db.collection('ai').document(doc.id)
            # 문서를 업데이트합니다.
            doc_ref.update({'signal': 1, 'ai_score': 1.0})
            
            # 여기서 필요한 처리를 수행합니다.
            # ...
            # 처리가 완료되면 signal을 2로 업데이트합니다.
            doc_ref.update({'signal': 2})
        
        # 시그널이 0인 문서가 없으면 루프를 다시 실행하기 전에 잠시 대기합니다.
        if not docs:
            print("No documents to update. Waiting...")
            time.sleep(10)  # 10초 대기

# 함수 호출
update_signal_and_score()
