#%%
from omegaconf import OmegaConf
from pathlib import Path
thermometer_path = Path(__file__).resolve().parents[2]
with open(thermometer_path.parent/'root/languageModels/api_key.txt') as f:
    # third line is the HG token
    HG_token = f.readlines()[2].strip()

#%%
cfg = OmegaConf.create()
cfg.root_path = str(thermometer_path)
cfg.training_seed = 0
cfg.seed_list = [0]
cfg.benchmark = 'mmlu'
cfg.test_dataset = 'computer_security'
cfg.thermometer_input_size = 2048
cfg.thermometer_hidden_size = 256
cfg.thermometer_lr = 1e-3
cfg.lambda_reg = 1e-2
cfg.Gamma_k = 1.25
cfg.Gamma_theta = 4.0
cfg.steps = 5000
cfg.model_type = 'decoder_only'
cfg.model_name = 'Llama-2-7b-chat-hf'
cfg.HG_token = HG_token
cfg.max_source_length = 256
cfg.max_target_length = 128
cfg.max_length = 2048
cfg.inference_batch_size = 2
cfg.train_batch_size = 128
cfg.eval_batch_size = 128

#%%
# save the configuration
