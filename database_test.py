import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Use the application default credentials
cred = credentials.Certificate('mykey')
firebase_admin.initialize_app(cred, {
  'projectId': 'danmoa-test',
})

db = firestore.client()

doc_ref = db.collection(u'users').document(u'user01')
doc_ref.set({
    u'level': 20,
    u'money': 700,
    u'job': "knight"
})
users_ref = db.collection(u'users')
docs = users_ref.stream()

for doc in docs:
    print(u'{} => {}'.format(doc.id, doc.to_dict()))
