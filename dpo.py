from unsloth import FastLanguageModel
import json
import os
import torch
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig


preference_dataset_path = "./dpo/dpo_preference_dataset_1000.jsonl"
sft_adapter_path = "./final_sft_adapters"

dataset = Dataset.from_json(preference_dataset_path)
# 필요한 경우 train/test로 분할할 수 있습니다. 여기서는 전체를 학습에 사용합니다.
dataset_dict = DatasetDict({'train': dataset})

# 2. Unsloth를 사용하여 모델 로드 및 설정
# 베이스 모델 ID
base_model_id = "unsloth/llama-3.2-1b-instruct-bnb-4bit"

# 4비트 양자화된 모델을 메모리에 효율적으로 로드하고 PEFT (LoRA) 설정을 준비합니다.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_id,
    max_seq_length=2048,  # 시퀀스 길이는 데이터에 맞춰 조절 가능
    dtype=None,           # None으로 두면 자동 설정
    load_in_4bit=True,    # 4비트 양자화 사용
)

# 베이스 모델 위에 기존에 학습한 SFT LoRA 어댑터를 로드합니다.
# 이 과정을 통해 모델은 SFT가 적용된 상태가 됩니다.
print(f"SFT 어댑터를 다음 경로에서 로드합니다: {sft_adapter_path}")
model.load_adapter(sft_adapter_path)
print("SFT 어댑터 로딩 완료.")

# 3. DPO 학습을 위한 PEFT(LoRA) 모델 설정
# 이제 SFT가 적용된 모델 위에 DPO 학습을 위한 새로운 LoRA 어댑터를 추가합니다.
# Unsloth는 이렇게 어댑터를 스태킹(stacking)하는 것을 지원합니다.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank (일반적으로 8, 16, 32, 64 중 선택)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# 4. DPO 학습 설정 및 실행
training_args = DPOConfig(
    beta=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=150,
    learning_rate=5e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    output_dir="final_dpo_adapter_original", # 결과 저장 경로
)

# DPOTrainer를 설정합니다.
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args, # DPOConfig 객체를 args에 전달
    train_dataset=dataset_dict["train"],
    tokenizer=tokenizer,
)

# DPO 학습 시작
dpo_trainer.train()

# 5. 학습된 모델 저장 (선택 사항)
# DPO 학습으로 생성된 LoRA 어댑터를 저장합니다.
# 이 어댑터는 SFT 어댑터 위에 추가된 변경 사항을 담고 있습니다.
model.save_pretrained("final_dpo_adapter_original")

print("DPO 학습이 완료되었습니다.")
print("학습된 DPO LoRA 어댑터는 'final_dpo_adapter_original' 폴더에 저장되었습니다.")
