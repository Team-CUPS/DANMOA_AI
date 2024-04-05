import pandas as pd

# 파일 경로 설정
file_path_1 = 'linkareer.csv'  # 첫 번째 CSV 파일 경로
file_path_2 = 'wanted.csv' # 두 번째 CSV 파일 경로
output_file_path = 'dev.csv'  # 출력될 CSV 파일 경로

# CSV 파일 불러오기, 열 이름 직접 지정
df1 = pd.read_csv(file_path_1, header=None, names=['sentence'], skiprows=1)  # 첫 번째 파일 불러오기, 첫 행 스킵
df2 = pd.read_csv(file_path_2, header=None, names=['sentence'], skiprows=1)  # 두 번째 파일 불러오기, 첫 행 스킵

# 모든 가능한 조합을 저장할 리스트 초기화
combinations = []

# 두 데이터프레임의 모든 가능한 조합 생성
for sentence1 in df1['sentence']:
    for sentence2 in df2['sentence']:
        combinations.append({'sentence1': sentence1, 'sentence2': sentence2, 'score': 0.5})

# 리스트를 DataFrame으로 변환
combined_df = pd.DataFrame(combinations)

# 결과를 새로운 CSV 파일로 저장
combined_df.to_csv(output_file_path, index=False)
