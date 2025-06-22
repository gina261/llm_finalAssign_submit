![initial](https://github.com/gina261/llm_finalAssign_submit/issues/1#issue-3166103031)

- 📄 **reward_model_with_lengthBias.py** : length bias를 갖는 reward model을 생성
- 📄 **dpo.py** : DPO(original) 학습 진행
- 📄 **dpo_mitigated.py** : DPO(mitigated) 학습 진행
- 📄 **generate_preference_data.py** : DPO 학습 위한 데이터셋 생성 (original Reward Model로 선호도 평가)
- 📄 **generate_preference_data_mitigated.py** : DPO 학습 위한 데이터셋 생성 (mitigated Reward Model로 선호도 평가)
- 📄 **merge_sft.py** : adapter load 중 오류 해결 위해 sft 먼저 merge하기 위한 코드
- 📄 **rm_feature_analysis_lengthBias_mitigate.ipynb** : length bias를 갖는 reward model로 실험 진행
- 📑 **dpo_test.ipynb** : original, mitigated DPO 모델 추론 및 평가
- 📑 **dpo_dataset_analysis.ipynb** : generate_preference_data로 만든 데이터 길이 편향 분석

- debiased_score_weights_alpha0.9.pt : mitigated된 마지막 layer weight (alpha 0.9)
- debiased_score_weights.pt : mitigated된 마지막 layer weight (alpha 1.0)
- dpo_preference_dataset_1000.jsonl : DPO 학습 위해 생성된 데이터셋 (1000개, original)
- dpo_preference_dataset.jsonl : DPO 학습 위해 생성된 데이터셋 (100개, original)
- dpo_preference_dataset_mitigated1000.jsonl : DPO 학습 위해 생성된 데이터셋 (1000개, mitigated)
- dpo_preference_dataset_mitigated.jsonl : DPO 학습 위해 생성된 데이터셋 (100개, mitigated)


- 📁 **final_reward_adapters_length_bias** : length bias를 갖는 reward model adapter
- 📁 **final_sft_adapters** : sft adapter
