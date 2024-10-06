test_dataset_list=("high_school_european_history" "business_ethics" "clinical_knowledge" "medical_genetics" "high_school_us_history" "high_school_physics" "high_school_world_history" "virology"
"high_school_microeconomics" "econometrics" "college_computer_science" "high_school_biology" "abstract_algebra" "professional_accounting" "philosophy" "professional_medicine"
"nutrition" "global_facts" "machine_learning" "security_studies" "public_relations" "professional_psychology" "prehistory" "anatomy" "human_sexuality" "college_medicine" "high_school_government_and_politics"
"college_chemistry" "logical_fallacies" "high_school_geography" "elementary_mathematics" "human_aging" "college_mathematics" "high_school_psychology" "formal_logic" "high_school_statistics" "international_law"
"high_school_mathematics" "high_school_computer_science" "conceptual_physics" "miscellaneous" "high_school_chemistry" "marketing" "professional_law" "management" "college_physics" "jurisprudence" "world_religions"
"sociology" "us_foreign_policy" "high_school_macroeconomics" "computer_security" "moral_disputes" "moral_scenarios" "electrical_engineering" "astronomy" "college_biology")
for seed in {0..4}; do
  for test_dataset in "${test_dataset_list[@]}"; do
    python ./src/train_thermometer.py --benchmark mmlu --test_dataset $test_dataset --model_type decoder_only --model_name Llama-2-7b-chat-hf  --training_seed $seed
  done
done

test_dataset_list=("arithmetic" "bbq_lite_json" "cifar10_classification" "contextual_parametric_knowledge_conflicts" "color" "elementary_math_qa" "epistemic_reasoning" "fact_checker" "formal_fallacies_syllogisms_negation"
"goal_step_wikihow" "hyperbaton" "logical_fallacy_detection" "mnist_ascii" "movie_dialog_same_or_different" "play_dialog_same_or_different" "real_or_fake_text" "social_iqa" "strategyqa" "timedial" "tracking_shuffled_objects"
"vitaminc_fact_verification" "unit_conversion" "winowhy")
for seed in {0..4}; do
  for test_dataset in "${test_dataset_list[@]}"; do
    python ./src/train_thermometer.py --benchmark mmlu --test_dataset $test_dataset --model_type decoder_only --model_name Llama-2-7b-chat-hf  --training_seed $seed
  done
done

test_dataset_list=("SQuAD" "SearchQA" "NaturalQuestionsShort" "HotpotQA" "NewsQA" "TriviaQA-web" "BioASQ" "DROP" "DuoRC.ParaphraseRC" "RACE" "RelationExtraction" "TextbookQA")
for seed in {0..4}; do
  for test_dataset in "${test_dataset_list[@]}"; do
    python ./src/train_thermometer.py --benchmark mmlu --test_dataset $test_dataset --model_type decoder_only --model_name Llama-2-7b-chat-hf  --training_seed $seed
  done
done