# 채용공고 크롤링 데이터 전처리
import json
import pandas as pd
import re

# 입력 JSON 파일 경로
json_file_path = 'wanted.json'

# JSON 파일 로드
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 데이터 추출 및 전처리
sentences = []

for item in data:
    # 각 필드에 대해 처리
    mainduties = item.get('MAINDUTIES', '')
    qualification = item.get('QUALIFICATION', '')
    preferential = item.get('PREFERENTIAL', '')
    # 각 필드의 특수문자 제거
    pattern = "[^0-9가-힣a-zA-Z一-龥.,)(\"'-·:}{》《~/%\[\]’‘“”=〉〈><_ ]"
    
    sum_sentences = mainduties + qualification + preferential
    lines = [line for line in sum_sentences.split('\n') if line.strip() != '']
    for line in lines:
        flag = True
        for filter in ('채용', '전형', '기타', '근무', '서류', '고용', '소개', '안내', '사항', '기간', '[', ']'):
            if (filter in line):
                flag = False
                break
        if (flag == True):
            sentences.extend([re.sub(pattern, '', sentence.strip()) for sentence in re.split(pattern, line) if sentence.strip() != ''])

# 리스트를 DataFrame으로 변환
df = pd.DataFrame(sentences, columns=['sentence'])
# DataFrame을 CSV 파일로 저장
csv_file_path = 'wanted.csv'
df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

print(f"CSV 파일이 '{csv_file_path}'에 저장되었습니다.")