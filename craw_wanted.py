# 원티드 채용공고 크롤링 코드
import requests
import datetime
import json
 
def crawling_wanted_data(parsed_data, info_keys):
    result={}
    for key in info_keys:
        if key == 'COMPANY':   # 회사 이름
            result['COMPANY'] = parsed_data['job']['company']['name']  
        elif key == 'GROUP_INTRO':   # 회사 소개
            result['GROUP_INTRO'] = parsed_data['job']['detail']['intro'] 
        elif key == 'WORKING_AREA':  # 근무 지역
            result['WORKING_AREA'] = parsed_data['job']['address'].get('full_location')
            #result['WORKING_AREA'] = parsed_data['job']['address']['full_location']
        elif key == 'TITLE':  # 제목
            result['TITLE'] = parsed_data['job']['position']
        elif key == 'MAINDUTIES': # 주요업무
            result['MAINDUTIES'] = parsed_data['job']['detail']['main_tasks']
        elif key == 'QUALIFICATION':  # 자격요건
            result['QUALIFICATION'] = parsed_data['job']['detail']['requirements']
        elif key == 'PREFERENTIAL':  # 우대사항
            result['PREFERENTIAL'] = parsed_data['job']['detail']['preferred_points']
        elif key == 'BENEFITS':  # 혜택 및 복지
            result['BENEFITS'] = parsed_data['job']['detail']['benefits']
        elif key == 'RECRUIT_DEADLINE':  # 마감날짜 None = 상시
            result['RECRUIT_DEADLINE'] = parsed_data['job']['due_time']
        elif key == 'geo_location': # 근무지역 (위도,경도)
            geo_location = parsed_data['job']['address'].get('geo_location')
            if geo_location:
                result['LAT'] = geo_location['n_location']['lat']
                result['LNG'] = geo_location['n_location']['lng']
            else:
                result['LAT'] = None
                result['LNG'] = None
        elif key == 'JOB_POSTING_URL':  # 채용공고 링크
            base = "https://www.wanted.co.kr/wd/"
            result['JOB_POSTING_URL'] = f"{base}{parsed_data['job']['id']}"
        elif key == 'TITLE_IMG':  # 대표 이미지
            result['TITLE_IMG'] = parsed_data['job']['title_img']['origin']
        elif key == 'LOGO_IMG':  # 로고 이미지
            result['LOGO_IMG'] = parsed_data['job']['logo_img']['origin']
        elif key == 'JOB_GROUP':
                 if parsed_data['job'].get('category_tags') and parsed_data['job']['category_tags']:
                    result['JOB_GROUP'] = parsed_data['job']['category_tags'][0]['parent_id']
                 else:
                     result['JOB_GROUP'] = None
        elif key == 'ID':
            result['ID']=parsed_data['job']['id']
    return result

parsed_data = []
all_data = []
id_set = set()

timestamp = int(datetime.datetime.now().timestamp())
tag_type_ids = 10101
years = -1
limit = 100
offset = 0
# tag_type_ids: 
# 518 개발/ 507 경영-비즈니스 / 523 마케팅 / 511 디자인 / 530 영업 
# 510 고객서비스-리테일 / 524 미디어 / 513 엔지니어링-설계
# 517 HR / 959 게임 제작 / 508 금융 / 522 제조-생산 / 515 의료-제약-바이오 / 10101 교육 
# 532 물류-무역 / 10057 식-음료 / 521 법률-법집행기관 / 509 건설-시설 / 514 공공-복지

# limit -> 100이면 100개씩 데이터추출 (한번에 100밖에 불가)
# offset -> 0번째부터 추출

# 10000개 데이터 가져오기 
total_data_count = 100000

base_url = "https://www.wanted.co.kr/api/v4/jobs/"

has_data = True

for offset in range(0, total_data_count, limit):
    url = "https://www.wanted.co.kr/api/v4/jobs?{}&country=kr&tag_type_ids={}&job_sort=job.latest_order&locations=all&years={}&limit={}&offset={}".format(timestamp, tag_type_ids, years, limit, offset)
    parse_company = requests.get(url).json()
    
    if parse_company and 'data' in parse_company and parse_company['data']:
        recur_ids = [temp['id'] for temp in parse_company['data']]

        for recur_id in recur_ids:
            detail_url = f"{base_url}{recur_id}?{timestamp}"
            job_data = requests.get(detail_url).json()

            info_keys = ['COMPANY', 'GROUP_INTRO', 'WORKING_AREA', 'TITLE', 'MAINDUTIES', 'QUALIFICATION', 'PREFERENTIAL',
                         'BENEFITS', 'RECRUIT_DEADLINE', 'JOB_POSTING_URL', 'TITLE_IMG', 'LOGO_IMG', 'JOB_GROUP', 'ID']  # 필요한 정보 키
            crawled_data = crawling_wanted_data(job_data, info_keys)

           
            all_data.append(crawled_data)
            id_set.add(crawled_data['ID'])

    else:
        print("No data returned or empty data.")
        has_data = False  # 데이터가 없음을 표시
        break  # 더 이상 데이터가 없으므로 반복 중단
        



 # 데이터가 있는 경우에만 파일로 저장
with open('wanted.json', 'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)


'''
import csv
import json

# JSON 파일 읽기
with open('wanted_data.json', 'r', encoding='utf-8', newline='') as file:
    json_data = json.load(file)

# 모든 JSON 객체의 키를 모아 헤더 생성
headers = set()
for entry in json_data:
    headers.update(entry.keys())

headers = list(headers)

# CSV 파일에 데이터 쓰기
csv_file = 'wanted_data_518.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=headers, delimiter = '|')
    writer.writeheader()
    for entry in json_data:
        writer.writerow(entry)

print(f'{csv_file} 파일에 데이터 저장 완료')
'''