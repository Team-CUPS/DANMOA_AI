import sys
import os
import pandas as pd
import torch
from functools import partial
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from data.info import STSDatasetFeatures  # 필요한 정보를 올바르게 가져오는지 확인하세요.
from SimCSE.arguments import DataTrainingArguments

def sts_prepare_features(examples, tokenizer, data_args):
    # 점수(scores)를 개별적으로 처리
    scores = [round((score/5),1) if score is not None else 0 for score in examples[STSDatasetFeatures.SCORE.value]]
    
    tokenized_inputs = tokenizer(
        examples[STSDatasetFeatures.SENTENCE1.value],
        examples[STSDatasetFeatures.SENTENCE2.value],
        padding='max_length',
        max_length=data_args.max_seq_length,
        truncation=True,
        return_tensors='pt'
    )

    # 각 특성을 리스트의 리스트로 변환
    features = {
        'input_ids': tokenized_inputs['input_ids'].tolist(),
        'attention_mask': tokenized_inputs['attention_mask'].tolist(),
        'token_type_ids': tokenized_inputs['token_type_ids'].tolist(),
        'labels': scores  # 개별 점수 리스트; 각 샘플에 대한 점수
    }

    return features

def main(model_name_or_path, dev_file, save_dir):
    data_args = DataTrainingArguments(
        dev_file=dev_file,
        save_dir=save_dir,
        preprocessing_num_workers=4,
        overwrite_cache=True,
        max_seq_length=128  # 최대 시퀀스 길이 설정
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # 데이터셋 로드
    if dev_file is not None:
        dataset = load_dataset('csv', data_files={'dev': dev_file}, delimiter=",")
        print("Dev data loaded successfully.")
    else:
        print("No dev data provided.")
        return

    prepare_features_with_param = partial(sts_prepare_features, tokenizer=tokenizer, data_args=data_args)

    # 데이터셋 전처리
    dataset = dataset.map(
        prepare_features_with_param,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=dataset["dev"].column_names,
        load_from_cache_file=False,
    )

    dataset.save_to_disk(data_args.save_dir)
    print("Processed dataset saved to:", data_args.save_dir)

if __name__ == "__main__":
    model_name_or_path = "kazma1/simcse-robertsmall-matching"  #kazma1/unsupervise_bert_base,kazma1/unsuperivse_roberta_large,kazma1/unsupervise_roberta_base,kazma1/unsupervise_roberta_small
    dev_file = "bootstrap_train.csv"  # 실제 dev 파일 경로로 수정 필요
    save_dir = "data/sts/bootstrap"

    main(model_name_or_path, dev_file, save_dir)
    print("Dataset processing completed successfully.")
