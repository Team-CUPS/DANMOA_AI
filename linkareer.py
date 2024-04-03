# 링커리어 자소서 데이터 크롤링 코드
import requests
import json
import urllib.parse

# 변수 설정 및 인코딩
json_variable = '{"filterBy":{"types":["ALL"],"status":"PUBLISHED"},"orderBy":{"field":"PASSED_AT","direction":"DESC"},"pagination":{"page":1,"pageSize":20}}&extensions={"persistedQuery":{"version":1,"sha256Hash":"9de3e00d7c080f21a562200ff07a8f380c724477caa7d564d356f47d8c84eb5b"}}'
encoded_str = urllib.parse.quote(json_variable, safe='&=')

# 요청 URL 설정
url = f'https://api.linkareer.com/graphql?operationName=CoverLetterList&variables={encoded_str}'

# 요청 보내기
response = requests.get(url)

# 응답을 JSON 형식으로 변환
data = response.json()

# JSON 파일로 저장
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("JSON 파일이 성공적으로 저장되었습니다.")