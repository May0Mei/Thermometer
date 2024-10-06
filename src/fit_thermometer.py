import os
import torch
from torch import optim
from torch import nn
import numpy as np
from tqdm import tqdm
import math
import random

from src.loss import GammaLoss, _ECELoss, Top_ECELoss, BrierLoss
from src.utils import compute_accuracy

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
TQDM_DISABLE = False

'''
Thermometer Architecture
'''
class Thermometer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Thermometer, self).__init__()
        self.relu = nn.ReLU()
        self.fc1_1 = nn.Linear(input_size, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, 1)

        self.fc2_1 = nn.Linear(input_size, hidden_size)
        self.fc2_2 = nn.Linear(hidden_size, 1)

        self.fc3_1 = nn.Linear(input_size, hidden_size)
        self.fc3_2 = nn.Linear(hidden_size, 1)

        self.fc_final = nn.Linear(3, 1)

    def forward(self, x):
        x_1 = self.fc1_1(x)
        x_1 = self.relu(x_1)
        x_1 = self.fc1_2(x_1)

        x_2 = self.fc2_1(x)
        x_2 = self.relu(x_2)
        x_2 = self.fc2_2(x_2)

        x_3 = self.fc3_1(x)
        x_3 = self.relu(x_3)
        x_3 = self.fc3_2(x_3)

        output = self.fc_final(torch.cat((x_1, x_2, x_3),1))
        return output

'''
The function to extract logits, labels, and features
'''
def get_logits_labels(model_type, model, device, tokenizer, vocabulary, dataloader):
    model.eval()
    # disable the gradient computation
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    # encode vocab
    vocab = list(vocabulary.values())
    encoded_vocab = [torch.LongTensor(tokenizer.encode(v, add_special_tokens=False)).to(device) for v in vocab]
    # collect all the LLM logits, true labels, and features of the dataset
    labels = []
    logits_list = [[] for v in encoded_vocab]
    feature_list = []
    for step, batch in enumerate(tqdm(dataloader, disable=TQDM_DISABLE)):
        batch_labels = np.where(batch['target_ids'] != -100, batch['target_ids'], tokenizer.pad_token_id)
        labels += tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
        source_ids = batch['source_ids'].to(device)
        source_mask = batch['source_mask'].to(device)
        if model_type == 'encoder_decoder':
            for i, ev in enumerate(encoded_vocab):
                possible_labels = torch.tile(ev, (source_ids.shape[0], 1))
                logits = model(input_ids=source_ids,
                            attention_mask=source_mask,
                            labels=possible_labels,
                            return_dict=True
                            ).logits[:, :, ev]
                logits_list[i].append(logits.cpu().detach().numpy())

            possible_labels = torch.tile(encoded_vocab[0], (source_ids.shape[0], 1))
            last_hidden_state = model(input_ids=source_ids,
                                     attention_mask=source_mask,
                                     labels=possible_labels,
                                     output_hidden_states=True,
                                     return_dict=True
                                     ).decoder_hidden_states[-1]
            feature_list.append(last_hidden_state.cpu().detach().numpy())
        elif model_type == 'decoder_only':
            for j in range(source_ids.size()[0]): # process one data per time to save memory
                outputs = model(input_ids=source_ids[j].unsqueeze(axis=0),
                                attention_mask=source_mask[j].unsqueeze(axis=0),
                                output_hidden_states=True,
                                return_dict=True)
                logits = outputs.logits[:, -1, :]
                for i, ev in enumerate(encoded_vocab):
                    logits_list[i].append(logits[:, ev].cpu().detach().numpy())
                feature_list.append(outputs.hidden_states[-1][:, -1, :].cpu().detach().numpy())
        else:
            raise NotImplementedError

    # collect all true labels
    converted_labels = []
    for k, element in enumerate(labels):
        converted_labels.append(vocab.index(element))
    target_labels = torch.tensor(converted_labels)
    # collect all logits
    for i, ev in enumerate(encoded_vocab):
        logits_list[i] = np.concatenate(logits_list[i], axis=0)
    logits_all = np.dstack(logits_list).squeeze()
    logits = torch.tensor(logits_all)
    # collect all features
    features = torch.tensor(np.concatenate(feature_list, axis=0)).squeeze()

    return logits, target_labels, features


'''
Training Thermometer
'''
def train_thermometer(
    model_thermometer,
    tasks_list,
    save_path,
    device,
    tokenizer,
    data_loader_dict,
    lambda_reg = 1e-2,
    Gamma_k = 1.25,
    Gamma_theta = 4.0,
    learning_rate = 1e-3,
    num_steps = 5000
):
    nll_criterion = nn.CrossEntropyLoss().to(device)
    regularizer = GammaLoss(Gamma_k, Gamma_theta).to(device) #to ensure the mode of gamma distribution is around 1
    optimizer = torch.optim.AdamW(model_thermometer.parameters(), lr=learning_rate, weight_decay=1e-4)
    model_thermometer = model_thermometer.to(device)
    # Start training
    best_nll_score = math.inf
    for t in range(num_steps):
        model_thermometer.train()
        # Randomly sample a dataset each step
        dataset_name = random.sample(tasks_list, 1)[0]
        print('train on dataset:', dataset_name)
        data_loader = data_loader_dict[dataset_name]
        train_dataloader = data_loader['train_loader']
        # encode vocab
        vocab = list(data_loader['target_vocabulary'].values())
        encoded_vocab = [torch.LongTensor(tokenizer.encode(v,add_special_tokens=False)).to(device) for v in vocab]
        for iteration, batch in enumerate(train_dataloader):
            logits_all = batch['unscaled_logits'].to(device)
            last_hidden_state = batch['features'].to(device)
            # collect target labels
            batch_labels = np.where(batch['target_ids'] != -100, batch['target_ids'], tokenizer.pad_token_id)
            labels = tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
            converted_labels = []
            for k, element in enumerate(labels):
                converted_labels.append(vocab.index(element))
            target_labels = torch.tensor(converted_labels).to(device)
            # compute the task specific temperature
            temperature_inverse = nn.Softplus()(model_thermometer(last_hidden_state)) # 1/T_i
            temperature_global_inverse = torch.mean(temperature_inverse) #1/N \sum_i 1/T_i
            # optimize Thermometer
            optimizer.zero_grad()
            loss = nll_criterion(logits_all*temperature_global_inverse, target_labels) \
                   + lambda_reg * regularizer(1/temperature_global_inverse)
            loss.backward()
            optimizer.step()
            break  # only conduct one-step gradient update (randomly sample a batch of data each step)
        torch.cuda.empty_cache() # Save GPU memory
        train_loss = loss.detach().float().item()
        temperature = 1/temperature_global_inverse.item()
        print(f"{t=}: {train_loss=} {temperature=}")
        # Validation
        if (t + 1) % 50 == 0:
            nll_score_avg = 0
            for dataset_name in tasks_list:
                print('eval on dataset', dataset_name)
                data_loader = data_loader_dict[dataset_name]
                calibration_results = eval_thermometer(model_thermometer, device, tokenizer=tokenizer,
                                                       vocabulary=data_loader['target_vocabulary'],
                                                       eval_dataloader=data_loader['val_loader'])
                nll_score_avg += calibration_results['NLL']
            print('Average NLL Loss', nll_score_avg/len(tasks_list))
            if nll_score_avg < best_nll_score:
                best_nll_score = nll_score_avg
                print('Saving Thermometer Model...')
                torch.save(model_thermometer.state_dict(),  os.path.join(save_path, 'model_ckpt.t7'))


'''
Eval Thermometer's calibration performance
'''
def eval_thermometer(
    model_thermometer,
    device,
    tokenizer,
    vocabulary,
    eval_dataloader
):
    model_thermometer.eval()
    # calibration metrics
    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = _ECELoss().to(device)
    top_ece_criterion = Top_ECELoss().to(device)
    # encode vocab
    vocab = list(vocabulary.values())
    encoded_vocab = [torch.LongTensor(tokenizer.encode(v, add_special_tokens=False)).to(device) for v in vocab]
    # collect target labels
    labels = []
    logits_list = []
    temperature_list = []
    for step, batch in enumerate(tqdm(eval_dataloader, disable=TQDM_DISABLE)):
        batch_labels = np.where(batch['target_ids'] != -100, batch['target_ids'], tokenizer.pad_token_id)
        labels += tokenizer.batch_decode(batch_labels, skip_special_tokens=True)
        logits = batch['unscaled_logits']
        logits_list.append(logits.cpu().detach().numpy())
        last_hidden_state = batch['features'].to(device)
        temperature_inverse = nn.Softplus()(model_thermometer(last_hidden_state))
        temperature_list.append(temperature_inverse.cpu().detach().numpy())
    # compute the task specific temperature
    temperature_inverse = np.concatenate(temperature_list, axis=0) # 1/T_i
    temperature_global_inverse = np.mean(temperature_inverse) #1/N \sum_i 1/T_i
    # collect all true labels
    converted_labels = []
    for k, element in enumerate(labels):
        converted_labels.append(vocab.index(element))
    target_labels = torch.tensor(converted_labels).to(device)
    # scale the logits by predicted temperature
    logits_all = np.concatenate(logits_list, axis=0)
    logits_scaled = torch.tensor(logits_all*temperature_global_inverse).to(device)
    # compute accuracy
    # copmute accuracy
    Accuracy = compute_accuracy(logits_scaled, target_labels)
    # compute calibration score
    nll_score = nll_criterion(logits_scaled, target_labels).item()
    ece_score, mce_score = ece_criterion(logits_scaled, target_labels)
    top_ece_score = top_ece_criterion(logits_scaled, target_labels)
    brier_score = BrierLoss(logits_scaled, target_labels)
    print('Temperature: %.3f,  NLL: %.3f, ECE: %.3f, Top-ECE: %.3f, MCE: %.3f, Brier: %.3f'
          % (1/temperature_global_inverse.item(),  nll_score, ece_score, top_ece_score, mce_score, brier_score))

    return {
            'Accuracy': Accuracy,
            'Temperature': 1/temperature_global_inverse.item(),
            'NLL': nll_score,
            'ECE': ece_score,
            'Top-ECE': top_ece_score,
            'MCE': mce_score,
            'Brier': brier_score
    }