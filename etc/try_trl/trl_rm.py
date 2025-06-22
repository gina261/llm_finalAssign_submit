import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig

# ====================================================================
# 1. 모델, 토크나이저, 모든 설정 정의
# ====================================================================

model_id = "unsloth/llama-3.2-1b-instruct-bnb-4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    modules_to_save=["score"],
)

training_args = RewardConfig(
    output_dir="./results_reward_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_steps=500,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    eval_steps=100,
    push_to_hub=False,
    report_to="none",
    max_length=1024,
)

# ====================================================================
# 2. 모델 및 토크나이저 로드, 그리고 레이어 교체
# ====================================================================

print("💾 모델과 토크나이저를 로드합니다...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    num_labels=1,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# ==================== 최종 수정 코드 ====================
print("🔄 문제가 되는 'score' 레이어를 교체하고 올바른 디바이스로 이동합니다...")

in_features = model.score.in_features
new_score_layer = torch.nn.Linear(
    in_features,
    1,
    bias=False,
    dtype=bnb_config.bnb_4bit_compute_dtype
)

# 새로 만든 레이어를 모델의 나머지 부분과 동일한 디바이스(GPU)로 이동시킵니다.
model.score = new_score_layer.to(model.device) # <--- 이 한 줄이 추가되었습니다!

print("✅ 'score' 레이어 교체 및 디바이스 이동 완료!")
# ========================================================

model.config.use_cache = False
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# ====================================================================
# 3. 데이터셋 준비 및 4. 학습 (이하 완전 동일)
# ====================================================================
print("💾 보상 모델용 데이터셋을 로드하고 전처리합니다...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train").select(range(20000))

def preprocess_reward_dataset(examples):
    return {"chosen": examples["chosen"], "rejected": examples["rejected"]}

processed_dataset = dataset.map(preprocess_reward_dataset, batched=True, remove_columns=dataset.column_names)
processed_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = processed_dataset["train"], processed_dataset["test"]

print(f"✅ 총 {len(train_dataset)}개의 데이터로 보상 모델 학습을 시작합니다.")

print("🚀 RewardTrainer를 초기화합니다...")
trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
)

print("🚀 학습을 시작합니다...")
trainer.train()

print("💾 학습된 모델(어댑터)을 저장합니다...")
trainer.save_model("./final_reward_adapters_data20000")

print("🎉 학습이 성공적으로 완료되었습니다!")