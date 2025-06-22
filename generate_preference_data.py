import torch
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from tqdm import tqdm

# ====================================================================
# 1. 경로 및 설정 정의
# ====================================================================
# unsloth 모델을 계속 사용합니다.
base_model_id = "unsloth/llama-3.2-1b-instruct-bnb-4bit"
sft_adapter_path = "./final_sft_adapters"
reward_adapter_path = "./final_reward_adapters_length_bias"
output_dataset_path = "./dpo/dpo_preference_dataset_2000.jsonl"

NUM_SAMPLES = 1000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"사용 장치(Device): {device}")

# 4비트 양자화 설정
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

# SFT 모델 로드 (CausalLM은 'score' 레이어가 없어 문제가 없으므로, 안정적으로 로드)
print("💾 응답 생성을 위한 SFT 모델을 로드합니다...")
sft_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map={"": device} # device_map="auto" 대신 명시적으로 단일 장치 할당
)
sft_model = PeftModel.from_pretrained(sft_model, sft_adapter_path)
sft_model.eval()


# --- Reward 모델 로드 (핵심적인 '레이어 교체' 해결법 적용) ---
print("💾 Reward 모델 로드를 시작합니다...")

# 1. 베이스 모델을 우선 로드합니다. (아직 GPU로 보내지 않음)
base_reward_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    num_labels=1,
)

# 2. 문제가 되는 'score' 레이어를 깨끗한 새 레이어로 교체합니다.
print("🔄 문제가 되는 'score' 레이어를 교체합니다...")
in_features = base_reward_model.score.in_features
new_score_layer = torch.nn.Linear(
    in_features=in_features,
    out_features=1,
    bias=False,
    dtype=torch.float16
)
base_reward_model.score = new_score_layer
print("✅ 'score' 레이어 교체 완료!")

# 3. 레이어 교체가 완료된 모델을 GPU로 이동시킵니다.
print(f"🔄 수정된 Reward 베이스 모델을 {device}로 이동합니다...")
base_reward_model.to(device)

# 4. 안정화된 베이스 모델 위에 Reward 어댑터를 적용합니다.
print(f"🔄 '{reward_adapter_path}'의 어댑터를 적용합니다...")
reward_model = PeftModel.from_pretrained(base_reward_model, reward_adapter_path)
reward_model.eval()

print("🎉 모든 모델 로딩 및 수정 완료!")


# ====================================================================
# 3. 데이터셋 생성 (이제 정상적으로 동작해야 합니다)
# ====================================================================
print(f"💾 프롬프트 데이터셋을 로드하여 {NUM_SAMPLES}개의 선호도 쌍을 생성합니다...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train").select(range(50000, 50000+NUM_SAMPLES))

preference_data = []

generation_kwargs = {
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 128,
    "temperature": 0.7,
}

for sample in tqdm(dataset, desc="선호도 데이터셋 생성 중"):
    full_text = sample["chosen"]
    prompt_end_str = "\n\nAssistant:"
    prompt_start_index = full_text.find("Human:")
    prompt_end_index = full_text.find(prompt_end_str)
    if prompt_start_index == -1 or prompt_end_index == -1:
        continue
    prompt_text = full_text[:prompt_end_index + len(prompt_end_str)]

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    # 응답 생성
    response_1_ids = sft_model.generate(input_ids, **generation_kwargs)
    response_2_ids = sft_model.generate(input_ids, **generation_kwargs)

    response_1_text = tokenizer.decode(response_1_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    response_2_text = tokenizer.decode(response_2_ids[0][len(input_ids[0]):], skip_special_tokens=True)

    if response_1_text == response_2_text:
        continue

    full_text_1 = prompt_text + response_1_text
    full_text_2 = prompt_text + response_2_text

    # 두 응답 평가
    reward_inputs = tokenizer([full_text_1, full_text_2], padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)

    with torch.no_grad():
        scores = reward_model(**reward_inputs).logits.squeeze()

    # squeeze 결과가 단일 스칼라 텐서일 경우를 대비
    if scores.dim() == 0:
        scores = scores.unsqueeze(0)
        
    # 만약 배치 사이즈 1에서 두 개의 응답을 생성하여 점수가 하나만 나온 경우 (오류 방지)
    if len(scores) < 2:
        continue
        
    score_1, score_2 = scores[0].item(), scores[1].item()

    if score_1 > score_2:
        chosen_response = full_text_1
        rejected_response = full_text_2
    else:
        chosen_response = full_text_2
        rejected_response = full_text_1

    preference_data.append({
        "prompt": prompt_text,
        "chosen": chosen_response,
        "rejected": rejected_response
    })

df = pd.DataFrame(preference_data)
df.to_json(output_dataset_path, orient='records', lines=True)

print(f"✅ 총 {len(df)}개의 선호도 데이터 생성 완료! 파일: '{output_dataset_path}'")