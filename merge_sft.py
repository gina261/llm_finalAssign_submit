# merge_sft_manual.py
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- 설정 ---
BASE_MODEL_ID = "unsloth/llama-3.2-1b-instruct-bnb-4bit"
SFT_ADAPTER_PATH = os.path.abspath("./final_sft_adapters")
MERGED_SFT_MODEL_PATH = os.path.abspath("./sft_merged_model")

# --- 실행 로직 ---
if __name__ == '__main__':
    print(f"Loading base model: {BASE_MODEL_ID}")

    # 4-bit 양자화 설정을 포함하여 베이스 모델 로드
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    print(f"Loading SFT adapter from: {SFT_ADAPTER_PATH}")
    
    # PeftModel.from_pretrained를 사용하여 베이스 모델 위에 어댑터를 직접 로드
    # 이렇게 하면 확실하게 PeftModel 객체가 생성됩니다.
    sft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)

    print("Successfully loaded adapter. Now merging...")

    # PeftModel의 공식적인 merge_and_unload 함수를 호출
    merged_model = sft_model.merge_and_unload()

    print(f"Merge complete. Saving merged model and tokenizer to: {MERGED_SFT_MODEL_PATH}")

    # 병합된 모델과 토크나이저를 디스크에 저장
    merged_model.save_pretrained(MERGED_SFT_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_SFT_MODEL_PATH)

    print("\nManual SFT model merging and saving complete.")
    print("This time, the merge was successful. You should not see the 'Skipping Merge' warning.")