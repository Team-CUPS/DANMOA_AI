from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "kazma1/unsupervise_bert_base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 입력 텍스트
input_text = "This is a sample input text."

# 입력 텍스트를 토큰화하고 토큰 ID로 변환
inputs = tokenizer(input_text, return_tensors="pt")

# 입력 텐서의 크기 확인
input_shape = inputs['input_ids'].shape

print("입력 텐서의 크기:", input_shape)
