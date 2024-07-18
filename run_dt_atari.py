import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset
import pickle
import os
import matplotlib.pyplot as plt
import wandb
import time


TEST_MODE = True

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model_type', type=str, default='dt')
parser.add_argument('--game', type=str, default='Pong')
parser.add_argument('--batch_size', type=int, default=128) 
if TEST_MODE:
    parser.add_argument('--context_length', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--num_steps', type=int, default=1)
else:
    parser.add_argument('--context_length', type=int, default=30) # K=1, 30, 50  
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_steps', type=int, default=500000) # DQN-replay 1%
# 
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./clean/')
##
parser.add_argument('--train_type', type=str, default='clean', choices=['clean', 'gaus', 'shot', 'imp', 'spe', 'fgsm'])
parser.add_argument('--noise_rate', type=float, default=0.0) # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
##
parser.add_argument('--eval_type', type=str, default='clean', choices=['clean', 'gaus', 'shot', 'imp', 'spe', 'fgsm']) 
parser.add_argument('--log_to_wandb', '-w', type=bool, default=False) 
parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint')
args = parser.parse_args()

set_seed(args.seed)

# wandb 
game_name, seed = args.game, args.seed
####
train_type = args.train_type
noise_rate = args.noise_rate
eval_type = args.eval_type
###
#data_dir = args.data_dir_prefix
#data_dir = data_dir.split('/')[-2]
###
model_type = args.model_type
group_name = f'{game_name}-{seed}-{train_type}-{noise_rate}-{eval_type}-{model_type}'
exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

# start time 
start_time = time.time()

if args.log_to_wandb:
    wandb.init(
        name=exp_prefix,
        group=group_name,
        project=f'{seed}-{train_type}-{noise_rate}-{eval_type}-{model_type}',
    )
    wandb.config.update(vars(args))


# log info
log_info = {
    'game': args.game, 'model_type': args.model_type, 
    'seed': args.seed, 'batch_size': args.batch_size, 
    'context_length': args.context_length, 'epochs': args.epochs, 
    'num_steps': args.num_steps, 'num_buffers': args.num_buffers, 
    'trajectories_per_buffer': args.trajectories_per_buffer, 
    'data_dir_prefix': args.data_dir_prefix, 'log_to_wandb': args.log_to_wandb
}


class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

# create dataset
obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer, args.train_type, args.noise_rate)

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

print('------------finish creating train dataset------------')
train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
model = GPT(mconf)

# display vocab size and block size
print("vocab_size",train_dataset.vocab_size)
print("block_size",train_dataset.block_size)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps), eval_type=args.eval_type)



trainer = Trainer(model, train_dataset, None, tconf, log_info) # initialize a trainer 

print('------------training start------------')
# eval_returns and eval_stds and T_rewards_list
eval_returns, eval_stds, T_rewards_list = trainer.train()
print('------------training end------------')

'''
# gamer normalized score function
def gamer_normalized_score(game_name, flat_T_rewards_list):
    # random score
    random_breakout_score = 2
    random_qbert_score = 164
    random_pong_score = -21
    random_seaquest_score = 68

    # gamer score
    gamer_breakout_score = 30
    gamer_qbert_score = 13455
    gamer_pong_score = 15
    gamer_seaquest_score = 42055

    # gamer normalized score
    if game_name == 'Breakout':
        gamer_normalized_score = ((flat_T_rewards_list - random_breakout_score) / abs(gamer_breakout_score - random_breakout_score)) * 100
    elif game_name == 'Qbert':
        gamer_normalized_score = ((flat_T_rewards_list - random_qbert_score) / abs(gamer_qbert_score - random_qbert_score)) * 100
    elif game_name == 'Pong':
        gamer_normalized_score = ((flat_T_rewards_list - random_pong_score) / abs(gamer_pong_score - random_pong_score)) * 100
    elif game_name == 'Seaquest':
        gamer_normalized_score = ((flat_T_rewards_list - random_seaquest_score) / abs(gamer_seaquest_score - random_seaquest_score)) * 100

    return gamer_normalized_score
'''
##########################################################################

# gamer normalized score function
def gamer_normalized_scores(game_name, flat_T_rewards_list):
    # 各ゲームのランダムスコアとゲーマースコアを定義
    random_scores = {
        'Breakout': 2,
        'Qbert': 164,
        'Pong': -21,
        'Seaquest': 68
    }
    gamer_scores = {
        'Breakout': 30,
        'Qbert': 13455,
        'Pong': 15,
        'Seaquest': 42055
    }

    # 正規化されたスコアを計算する関数
    def calculate_normalized_score(score):
        return ((score - random_scores[game_name]) / abs(gamer_scores[game_name] - random_scores[game_name])) * 100

    # リスト内の各スコアに対して正規化されたスコアを計算
    normalized_scores = [calculate_normalized_score(score) for score in flat_T_rewards_list]
    return normalized_scores
##########################################################################

# result
print('------------result------------')
print("T_rewards_list:", T_rewards_list)

# calculate total, average, std of T_rewards_list
flat_T_rewards_list = np.array(T_rewards_list).flatten()
total_T_rewards_list = np.sum(flat_T_rewards_list)
average_T_rewards_list = np.mean(flat_T_rewards_list)
std_T_rewards_list = np.std(flat_T_rewards_list)

# display total, average, std of T_rewards_list
#print("flat of T rewards list:", flat_T_rewards_list)
print("total of T rewards list:", total_T_rewards_list)
print("average of T rewards list:", average_T_rewards_list)
print("std of T rewards list:", std_T_rewards_list)

# display average +- std of T_rewards_list
print("average of T rewards list ± std of T rewards list:", average_T_rewards_list, "±", std_T_rewards_list)
#print("gamer normalized score:", gamer_normalized_score(args.game, average_T_rewards_list))

# display average +- std of T_rewards_list in gamer normalized score
gamer_normalized_score_list = gamer_normalized_scores(args.game, flat_T_rewards_list)
gamer_normalized_score_average = np.mean(gamer_normalized_score_list)
gamer_normalized_score_std = np.std(gamer_normalized_score_list)
print("gamer normalized score list:", gamer_normalized_score_list)
print("average ± std in gamer normalized score:", gamer_normalized_score_average, "±", gamer_normalized_score_std)

##########################################################################
#print("eval_returns:", eval_returns)
#print("eval_stds:", eval_stds)
##########################################################################
print('------------result end------------')


# eval_returns to array and eval_stds to array
eval_returns_array = np.array(eval_returns)
eval_stds_array = np.array(eval_stds)

# eval_returns_array in solid line and eval_stds_array in shadow
# create x_axis_data
x_axis_data = np.arange(1, len(eval_returns_array) + 1)

# create figure
plt.figure(figsize=(10, 5))  
plt.grid(True)
plt.plot(x_axis_data, eval_returns_array, label='eval return', marker='o',)  
plt.fill_between(x_axis_data, 
                 eval_returns_array - eval_stds_array, 
                 eval_returns_array + eval_stds_array, 
                 alpha=0.2, color='cornflowerblue', label='eval std')  

# x-axis and y-axis setting
plt.xticks(x_axis_data)  
#plt.ylim(bottom=0)   # if eval_return or eval_std is not negative, use this

# label setting
plt.xlabel('epoch')
plt.ylabel('eval average return')
plt.title(f"{args.game} {args.model_type} {args.seed}", fontsize=14)
plt.legend()

# save figure
plt.tight_layout()
plt.savefig(f'result_fig/{args.game}_{args.model_type}_{args.seed}_eval.png')
#plt.show()

# figure to wandb
filename = f'result_fig/{args.game}_{args.model_type}_{args.seed}_eval.png'
if args.log_to_wandb:
    wandb.log({'eval average return fig': wandb.Image(filename)})
##########################################################################

# end time
end_time = time.time()
total_time = end_time - start_time
# total time 
print("------------total time------------")
#print("total time by second:", total_time, "s")
#print("total time by minute:", total_time / 60, "min")
print("total time by hour:", total_time / 3600, "h")
print("------------total time end------------")

# total time to wandb
if args.log_to_wandb:
    wandb.log({'total time': total_time})
##########################################################################

##########################################################################
'''
# 以降のモデルロードに関するコードは実験後、削除する
# 学習終了後にモデルを保存
model_save_path = 'trained_model_weights/{}_{}_{}.pth'.format(args.game, args.model_type, args.seed)
torch.save(model.state_dict(), model_save_path)
'''