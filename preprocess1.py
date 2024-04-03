# 자소서 크롤링 데이터 전처리
import json
import pandas as pd
import re

# JSON 파일 경로
json_file_path = 'data.json'

# JSON 파일 로드
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 숫자 다음에 오는 점을 제외하고 문장을 분리하기 위한 정규 표현식 패턴
pattern = r'(?<!\d)\.(?!\d)'

# content 텍스트 추출 및 처리
sentences = []
for edge in data['data']['coverLetters']['edges']:
    content = edge['node']['content']
    # '\n'을 기준으로 나누고, 빈 줄을 제외한 문장만 추가
    lines = [line for line in content.split('\n') if line.strip() != '']
    for line in lines:
        # '.'을 기준으로 문장을 나누고, 빈 문장을 제외한 문장 중 특수문자를 필터링해서 추가
        sentences.extend([re.sub("[^0-9가-힣a-zA-Z一-龥.,)(\"'-·:}{》《~/%\[\]’‘“”=〉〈><_ ]", '', sentence.strip()) for sentence in re.split(pattern, line) if sentence.strip() != ''])
# 리스트를 DataFrame으로 변환
df = pd.DataFrame(sentences, columns=['Sentence'])

# DataFrame을 CSV 파일로 저장
csv_file_path = 'content.csv'
df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

print(f"CSV 파일이 '{csv_file_path}'에 저장되었습니다.")