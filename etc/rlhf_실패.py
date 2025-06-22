import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
# Value Head가 포함된 모델을 다시 사용합니다. 이 구조의 핵심입니다.
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, PeftModel
from tqdm import tqdm

# ====================================================================
# 1. 모든 경로 및 설정 정의
# ====================================================================

# SFT 어댑터 경로
sft_adapter_path = "./final_sft_adapter"
# -------------------------

base_model_id = "unsloth/llama-3.2-1b-instruct-bnb-4bit"
reward_adapter_path = "./final_reward_adapters_length_bias"
rlhf_adapter_path = "./final_rlhf_adapter_length_bias" # 최종적으로 저장될 RLHF 어댑터 경로

# PPO 학습을 위한 설정
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    kl_coef=0.1,
    num_ppo_epochs=4,
    seed=42,
)

# RLHF 과정에서 Policy 모델을 튜닝하기 위한 LoRA 설정
# SFT 때 사용했던 설정과 동일하게 맞추는 것이 일반적입니다.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM", # RLHF는 Causal LM이므로 "CAUSAL_LM"으로 설정
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

# 4-bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ====================================================================
# 2. 모델 및 토크나이저 로드
# ====================================================================
print("💾 토크나이저를 로드합니다...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("💾 학습 대상 Policy 모델(Actor)을 로드합니다...")
policy_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
policy_model = PeftModel(policy_model, lora_config)

print("💾 참조(Ref) 모델을 로드합니다...")
ref_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
ref_model = PeftModel(ref_model, lora_config)

print("💾 가치(Value) 모델(Critic)을 로드합니다...")
value_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
value_model = PeftModel(value_model, lora_config)

print("💾 평가용 Reward 모델을 로드합니다...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    num_labels=1,
)
reward_model = PeftModel.from_pretrained(reward_model, reward_adapter_path)

# [최종 수정] LoRA 어댑터를 원본 모델에 병합하고 PEFT 래퍼를 제거합니다.
print("✨ Reward 모델의 LoRA 어댑터를 병합합니다...")
reward_model = reward_model.merge_and_unload()
print("✅ 병합 완료!")


reward_model.eval()
print("✅ 모든 모델 로드 완료!")


# ====================================================================
# 3. 데이터셋 준비 (RLHF용 프롬프트)
# ====================================================================
print("💾 RLHF용 프롬프트 데이터셋을 준비합니다...")
def create_prompt(sample):
    prompt_start = "\n\nHuman: "
    prompt_end = "\n\nAssistant:"
    full_text = sample["chosen"]
    start_index = full_text.find(prompt_start)
    end_index = full_text.find(prompt_end, start_index)
    if start_index != -1 and end_index != -1:
        prompt = full_text[start_index + len(prompt_start):end_index].strip()
        sample["query"] = f"{prompt_start}{prompt}{prompt_end}"
    else:
        sample["query"] = ""
    return sample

dataset = load_dataset("Anthropic/hh-rlhf", split="train").select(range(50000, 50100))
dataset = dataset.map(create_prompt).filter(lambda x: len(x["query"]) > 0)

# PPOTrainer 내부에서 사용할 토크나이징 함수
def tokenize_function(examples):
    return tokenizer(examples["query"], truncation=True)

print("✨ 데이터셋을 토크나이징하고 정리합니다...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    # [최종 수정] 'input_ids'와 'attention_mask'를 제외한 모든 원본 컬럼을 명시적으로 삭제합니다.
    remove_columns=['chosen', 'rejected']
)
tokenized_dataset.set_format("torch")
print("✅ 토크나이징 및 정리 완료!")

# ====================================================================
# 4. PPOTrainer 초기화 및 학습
# ====================================================================
# [새로운 코드 추가] 사용자 정의 데이터 콜레이터 정의
class DataCollatorForPPO:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        # 'query' 문자열을 분리합니다.
        # .pop()은 딕셔너리에서 해당 키를 제거하고 그 값을 반환합니다.
        queries = [feature.pop("query") for feature in features]
        
        # 나머지 숫자 데이터(input_ids, attention_mask)만 기본 콜레이터로 처리합니다.
        batch = self.base_collator(features)
        
        # 분리했던 'query'를 다시 배치에 추가합니다.
        batch["query"] = queries
        
        return batch

print("🚀 PPOTrainer를 초기화합니다...")

data_collator = DataCollatorForPPO(DataCollatorWithPadding(tokenizer=tokenizer))

# PPOTrainer는 최소한의 요소만 받습니다.
ppo_trainer = PPOTrainer(
    args=ppo_config,              # 'config'가 아닌 'args' 사용
    model=policy_model,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    processing_class=tokenizer,   # 'tokenizer'가 아닌 'processing_class' 사용
    train_dataset=tokenized_dataset, # 'dataset'이 아닌 'train_dataset' 사용 및 토크나이징된 데이터 전달
    data_collator=data_collator,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 128,
}

print("🚀 RLHF 학습을 시작합니다 (명시적 루프 사용)...")

for epoch in range(ppo_config.num_ppo_epochs):
    print(f"Epoch {epoch+1}/{ppo_config.num_ppo_epochs}")
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        # 1. [최종 수정] policy_model에서 직접 응답 생성
        #    이것이 가장 확실하고 표준적인 방법입니다.
        #    생성 결과는 '프롬프트 + 응답' 텐서입니다.
        response_tensors = policy_model.generate(query_tensors, **generation_kwargs)

        # 2. [최종 수정] 보상 계산을 위해 순수 '응답' 부분만 분리
        #    response_tensors에서 query_tensors 부분을 잘라냅니다.
        batch["response"] = [
            tokenizer.decode(response[len(query):], skip_special_tokens=True)
            for query, response in zip(query_tensors, response_tensors)
        ]
        
        # 이제 texts_for_reward는 '프롬프트(query_text) + 응답(response_text)'로 올바르게 조합됩니다.
        texts_for_reward = [q + r for q, r in zip(batch['query'], batch['response'])]
        
        reward_inputs = tokenizer(texts_for_reward, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(reward_model.device)
        
        with torch.no_grad():
            rewards = reward_model(**reward_inputs).logits.squeeze(-1)
            
        # 3. PPO 업데이트 단계 수행
        #    step 함수에는 '프롬프트'와 '프롬프트+응답' 텐서를 그대로 전달합니다.
        #    PPOTrainer는 내부적으로 이 둘을 비교하여 KL 페널티 등을 계산합니다.
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards.cpu().numpy())


print("✅ RLHF 학습 완료!")

# ====================================================================
# 5. 최종 모델 저장
# ====================================================================
print(f"💾 최종 RLHF 어댑터를 '{rlhf_adapter_path}'에 저장합니다...")
ppo_trainer.save_pretrained(rlhf_adapter_path)
print("🎉 모든 과정이 성공적으로 완료되었습니다!")