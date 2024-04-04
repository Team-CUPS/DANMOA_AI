from datasets import load_from_disk


# 데이터셋 로드 (데이터셋 이름과 경로는 실제 상황에 맞게 조정하세요)
dataset = load_from_disk('data/datasets/train')

# 처음 몇 개의 샘플 출력

print(dataset[:1])
