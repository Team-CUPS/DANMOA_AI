from datasets import load_from_disk

# 데이터셋 로드
dataset = load_from_disk("data/sts/eval/dev")

# 데이터셋의 첫 번째 항목의 키 값 출력
first_item_keys = dataset[0].keys()
print(f"Available keys in the dataset: {list(first_item_keys)}\n")

# 첫 번째 샘플에서 필드의 구체적인 값 확인
sample = dataset[0]
print(f"Sample 0: Input IDs = {sample['input_ids']}")
print(f"Sample 0: Attention Mask = {sample['attention_mask']}")
if 'token_type_ids' in sample:
    print(f"Sample 0: Token Type IDs = {sample['token_type_ids']}")
print(f"Sample 0: Labels = {sample['labels']}")

# 첫 번째 5개의 샘플에 대한 정보를 확인
for i in range(5):
    sample = dataset[i]
    print(f"Sample {i}: Input IDs = {sample['input_ids'][:5]}...")  # 처음 5개의 값만 출력하여 확인
    print(f"Sample {i}: Attention Mask = {sample['attention_mask'][:5]}...")
    if 'token_type_ids' in sample:
        print(f"Sample {i}: Token Type IDs = {sample['token_type_ids'][:5]}...")
    print(f"Sample {i}: Labels = {sample['labels']}\n")

