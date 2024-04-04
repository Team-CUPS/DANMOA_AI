import requests
import json
import urllib.parse

# JSON 변수 정의 및 인코딩
json_variable = '{"filterBy":{"organizationName":"","role":"개발자","keyword":"","types":["ALL"],"status":"PUBLISHED"},"orderBy":{"field":"RELEVANCE","direction":"DESC"},"pagination":{"page":1,"pageSize":20}}&extensions={"persistedQuery":{"version":1,"sha256Hash":"9de3e00d7c080f21a562200ff07a8f380c724477caa7d564d356f47d8c84eb5b"}}'
encoded_str = urllib.parse.quote(json_variable, safe='&=')

# 요청 URL 구성
url = f'https://api.linkareer.com/graphql?operationName=CoverLetterList&variables={encoded_str}'

# API 요청 및 응답 수신
response = requests.get(url)

# 응답 데이터를 JSON 파일로 저장
if response.status_code == 200:
    data = response.json()
    with open('linkareer.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print("응답 데이터가 'response_data.json' 파일에 저장되었습니다.")
else:
    print(f"요청 실패: 상태 코드 {response.status_code}")