import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


# 1. 모델 및 토크나이저 설정
# Unsloth에서 제공하는 4비트 양자화된 Llama-3.2-1B 모델을 사용합니다.
# 공식 모델이 나오면 해당 ID로 변경할 수 있습니다.
model_id = "unsloth/llama-3.2-1b-instruct-bnb-4bit"

# 4비트 양자화 설정 (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Llama3.2 모델의 어텐션 및 피드포워드 레이어를 타겟으로 지정
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)


# 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False # 학습 시에는 캐시 비활성화

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # PAD 토큰 설정
    
# 2. 데이터셋 로드 및 전처리 (비율 조정 적용)

# =================== 비율 설정 ===================
TARGET_TOTAL_SAMPLES = 20000  # 학습에 사용할 총 데이터 수
ALPACA_RATIO = 0.7            # Alpaca 데이터셋의 비율 (0.0 ~ 1.0)
GSM8K_RATIO = 1.0 - ALPACA_RATIO
# ===============================================

def preprocess_alpaca(examples):
    # ... (이전 코드와 동일)
    processed_examples = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
        processed_examples.append({"messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": output}]})
    return {"messages": [example["messages"] for example in processed_examples]}

def preprocess_gsm8k(examples):
    # ... (이전 코드와 동일)
    processed_examples = []
    for question, answer in zip(examples['question'], examples['answer']):
        processed_examples.append({"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]})
    return {"messages": [example["messages"] for example in processed_examples]}

# 데이터셋 로드
print("💾 데이터셋 로드를 시작합니다...")
alpaca_dataset = load_dataset("yahma/alpaca-cleaned", split="train")
gsm8k_dataset = load_dataset("gsm8k", "main", split="train")

print(f"원본 데이터 크기: Alpaca={len(alpaca_dataset)}, GSM8K={len(gsm8k_dataset)}")

# 각 데이터셋에서 샘플링할 개수 계산
num_alpaca_samples = int(TARGET_TOTAL_SAMPLES * ALPACA_RATIO)
num_gsm8k_samples = int(TARGET_TOTAL_SAMPLES * GSM8K_RATIO)

# 원본 데이터셋 크기보다 많이 샘플링할 수 없도록 조정
num_alpaca_samples = min(num_alpaca_samples, len(alpaca_dataset))
num_gsm8k_samples = min(num_gsm8k_samples, len(gsm8k_dataset))

print(f"샘플링할 데이터 크기: Alpaca={num_alpaca_samples}, GSM8K={num_gsm8k_samples}")

# 데이터셋을 섞고 필요한 만큼 샘플링
# .shuffle()을 먼저 해야 무작위 샘플을 얻을 수 있습니다.
alpaca_sampled = alpaca_dataset.shuffle(seed=42).select(range(num_alpaca_samples))
gsm8k_sampled = gsm8k_dataset.shuffle(seed=42).select(range(num_gsm8k_samples))

# 전처리 적용
print("🔄 데이터셋 전처리를 진행합니다...")
alpaca_processed = alpaca_sampled.map(lambda examples: preprocess_alpaca(examples), batched=True, remove_columns=alpaca_sampled.column_names)
gsm8k_processed = gsm8k_sampled.map(lambda examples: preprocess_gsm8k(examples), batched=True, remove_columns=gsm8k_sampled.column_names)

# 두 데이터셋 병합
print("🤝 데이터셋을 병합하고 셔플합니다...")
combined_dataset = concatenate_datasets([alpaca_processed, gsm8k_processed])
combined_dataset = combined_dataset.shuffle(seed=42)

print(f"✅ 총 {len(combined_dataset)}개의 데이터로 학습을 시작합니다.")
print("샘플 데이터:", combined_dataset[0]['messages'])


# 3. SFTTrainer 설정 및 학습
# 학습 인자(Arguments) 설정
training_args = TrainingArguments(
    output_dir="./results_sft_mixed",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=1000,  # 필요에 따라 총 학습 스텝 수 조정
    logging_steps=10,
    save_steps=100,
    fp16=True, # bfloat16을 쓸 수 없는 환경에서는 fp16 사용
    push_to_hub=False,
    report_to="none", # wandb 등 로깅 도구 사용 시 "wandb"로 변경
)

# SFTTrainer 초기화
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
    peft_config=lora_config,
)

# 학습 시작
print("🚀 SFT 학습을 시작합니다...")
trainer.train()

# 학습 완료 후 모델 저장
print("💾 학습된 모델(어댑터)을 저장합니다...")
trainer.save_model("./final_sft_adapters")

print("🎉 학습이 성공적으로 완료되었습니다!")