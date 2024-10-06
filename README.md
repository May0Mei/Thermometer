# Thermometer: Towards Universal Calibration for Large Language Models
#### This repository contains official implementation of the paper [Thermometer: Towards Universal Calibration for Large Language Models](https://arxiv.org/abs/2403.08819).

## Requirements
```bash
pip install torch==2.2.1 transformers==4.28.1 evaluate==0.4.1 tqdm pandas
```
## File Structure
#### *src* folder includes all the source code for the experiments.
- ### *configs* folder: 
    includes all the information of training configurations and model hyper-parameter.
- ### *data* folder:
    includes the dataloader to load and process the datasets.
- ### other code files:
  - *process_mrqa.py* pre-process the raw data in free-form QA datasets MRQA; 
  - *extract_features.py* aims to extract labels, features, and logits from pretrained LLMs;
  - *train_thermometer.py* and *eval_thermometer.py* contain the main function to train Thermometer,
  and the functions to evaluate calibration performance of trained Thermometer, respectively.


## Usage
#### We provide the scripts to help reproduce the results of our paper.
- ### Step-1: extract the features and logits from pre-trained LLMs,
    ```
    exract.sh
    ```
- ### Step-2: train Thermometer model,
    ```
    train.sh
    ```
- ### Step-3: evaluate calibration of trained Thermometer model,
    ```
    eval.sh
    ```
- ### Free-form QA task requires an additional step to pre-process the raw data, i.e., append the LLM's response to the prompts,
    ```
    mrqa.sh
    ```
- ### Choose different types of LLMs,
    ```
    --model_type decoder_only --model_name Llama-2-7b-chat-hf
    --model_type encoder_decoder --model_name flan-t5-xl
    ```

## Citation
#### If you find this repository helpful for your research, please consider citing our paper, 
```
@InProceedings{pmlr-v235-shen24c,
  title = {Thermometer: Towards Universal Calibration for Large Language Models},
  author =  {Shen, Maohao and Das, Subhro and Greenewald, Kristjan and Sattigeri, Prasanna and Wornell, Gregory W. and Ghosh, Soumya},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {44687--44711},
  year =  {2024},
  editor =  {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume =  {235},
  series =  {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR}
}
```
