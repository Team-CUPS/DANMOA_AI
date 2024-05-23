# DANMOA AI Recommend System

## Table of Contents
- [Overview](#overview)
- [Model Description](#model-description)
- [Developed by](#developed-by)
- [Model Details](#model-details)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)
- [Evaluation](#evaluation)
- [How to Get Started With the Model](#how-to-get-started-with-the-model)
- [How to Get Started With the Program](#how-to-get-started-with-the-program)
- [Program Structure](#program-structure)
- [Interacting with the Program: Input and Output Examples](#interacting-with-the-program-input-and-output-examples)

## Overview
- 사용자의 스터디 방향을 잡아주고 추천해주는 AI
- 자기소개서, 채용공고 데이터 활용
- RoBERTa 기반 모델 설계
- SimCSE 데이터 전처리

### Model Description
This LLM utilizes Masked Language Modeling (MLM) and employs the SimCSE technique with NLI and STS datasets for specialized downstream tasks. The model is designed to accurately match resumes with corresponding companies by analyzing textual similarities. Post-matching, it leverages a GPT-based framework to provide custom QA sessions for each company, enhancing the relevance and precision of interactions.


- **Developed by:** umhyeonho(umleeho1),abee3417
- **Model Type:** Fill-Mask + LLM + gpt 4.0
- **Language(s):** Korea
- **License:** [More Information needed]
- **Parent Model:** See the [klue/BERT base uncased model](https://huggingface.co/klue/bert-base)

## Uses

#### Direct Use

"This model is designed to recommend question and answer (QA) pairs based on user input."


## Training


#### Training Procedure
* **type_vocab_size:** 2
* **vocab_size:** 32000
* **num_hidden_layers:** 12


#### Training Data
unsupervised dataset: wanted.csv,linkareer.csv (자소서+채용공고)
nli dataset: eval.csv
sts_dataset: sts_train.csv

## Evaluation
| Model                         | Cosine Pearson | Cosine Spearman | Euclidean Pearson | Euclidean Spearman | Manhattan Pearson | Manhattan Spearman | Dot Pearson | Dot Spearman |
|-------------------------------|----------------|-----------------|-------------------|--------------------|-------------------|--------------------|-------------|--------------|
| SimCSE-RoBERTasmall-matching  | 70.23          | 67.34           | 66.32             | 66.28              | 63.44             | 61.52              | 59.08       | 60.08        |


#### Results

[More Information Needed]

## How to Get Started With the Model
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_path = "kazma1/simcse-robertsmall-matching"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
```
## How to Get Started With the Program
```python
import time
import firebase_admin
from firebase_admin import firestore, credentials
from transformers import AutoModel, AutoTokenizer
import torch
import openai
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# To start using the model, run the 'matching_firebase.py' script. This script integrates the model with Firebase for processing user inputs and generating responses. 
# Ensure all the necessary libraries are installed and properly configured before executing the script.
```

##Program Structure

This program is designed to handle multiple tasks concurrently using a ThreadPoolExecutor with 10 threads. This setup allows efficient management of various operations such as fetching data, processing user inputs, and interacting with the model. Additionally, the program utilizes the ChatGPT API to transform user inputs into a question and answer (QA) format, enhancing the interaction by providing contextually relevant and coherent responses.

##interacting with the Program: Input and Output Examples
input ex) "백엔드 개발자 스터디로 모였는데 무슨 공부를 하는게 좋을까?"
ouput ex) "백엔드 개발자 스터디를 시작하셨다면, 다음과 같은 기술 스택을 공부하는 것이 좋습니다. Spring과 Spring Boot를 활용한 웹서비스 설계 및 개발, JPA와 Hibernate를 사용한 ORM 및 도메인 모델링, Restful API 설계 및 개발, 그리고 AWS 환경에서의 개발 및 운영 경험을 쌓는 것이 중요합니다. 또한, 기본적인 Linux/Unix 명령어 사용 능력과 함께 MySQL 등의 RDBMS 경험, 빌드/테스트/배포 자동화, 그리고 통계 배치 개발 경험을 쌓는 것도 유익할 것입니다."


