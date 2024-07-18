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
from fixed_replay_buffer import FixedReplayBuffer
import skimage as sk


#ノイズ関数を定義
def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    orig_dtype = x.dtype  # 元のデータ型を保存
    x = np.array(x, dtype=np.float32) / 255.  # データ型をfloat32に変換
    x_noisy = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)
    return (x_noisy * 255).astype(orig_dtype)  # 元のデータ型に戻して返す

def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    orig_dtype = x.dtype  # 入力の元のデータ型を保存する
    x = np.array(x, dtype=np.float32) / 255.  # 計算のためにfloat32に変換
    x_noisy = np.clip(np.random.poisson(x * c) / float(c), 0, 1)
    return (x_noisy * 255).astype(orig_dtype)  # 元のデータ型に戻す

def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]  # ノイズ強度レベル
    orig_dtype = x.dtype  # 入力の元のデータ型を保存する
    x = sk.util.random_noise(np.array(x, dtype=np.float32) / 255., mode='s&p', amount=c)  # ソルト&ペッパーノイズを適用
    return (np.clip(x, 0, 1) * 255).astype(orig_dtype)  # 値をクリップし、元の範囲と型に戻す

def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]  # ノイズ強度レベル
    orig_dtype = x.dtype  # 入力の元のデータ型を保存する
    x = np.array(x, dtype=np.float32) / 255.  # float32に変換して正規化する
    x_noisy = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1)
    return (x_noisy * 255).astype(orig_dtype)  # 値をクリップし、元の範囲と型に戻す

#fgsm関数を定義
def standardize(tensor):
    return (tensor - tensor.mean()) / tensor.std()

def unstandardize(tensor):
    return tensor * tensor.std() + tensor.mean()

def fgsm(x, model, severity=1):
    c = [8, 16, 32, 64, 128][severity - 1]
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).requires_grad_(True) / 255.0  # 正規化
    logits = model(x)
    model.zero_grad()
    loss = F.cross_entropy(logits, logits.data.max(1)[1].squeeze(), reduction='sum')
    loss.backward()
    x_adv = torch.clamp(unstandardize(x.data) + c / 255.0 * unstandardize(torch.sign(x.grad.data)), 0, 1).squeeze(0) * 255  # 元の範囲に戻す
    return x_adv.detach().cpu().numpy().astype(np.uint8)
######

#ノイズを適用する関数
def apply_noise(obss, train_type, noise_rate, severity=1, model=None):
    if train_type == 'clean':
        print(f"ノイズ適用なし: {len(obss)}件")
        return obss  # ノイズ適用なし

    num_samples = len(obss)
    num_noisy = int(num_samples * noise_rate)  # ノイズを適用する観測データの数
    noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)  # ノイズを適用するインデックスを重複なしでランダムに選択

    noise_functions = {
        'gaus': gaussian_noise,
        'shot': shot_noise,
        'imp': impulse_noise,
        'spe': speckle_noise,
        'fgsm': lambda x, severity: fgsm(x, model, severity) if model is not None else x
    }

    if train_type not in noise_functions:
        raise ValueError(f"Unsupported train type '{train_type}'. Supported types are: {list(noise_functions.keys())}")
    
    noise_func = noise_functions[train_type]  # ノイズ関数を直接取得

    # ノイズを適用
    for idx in noisy_indices:
        obss[idx] = noise_func(obss[idx], severity)

    # デバッグ情報の出力
    print(f"ノイズ適用数: {num_noisy}件, ノイズ未適用数: {num_samples - num_noisy}件 (合計: {num_samples}件, ノイズ適用率: {noise_rate * 100:.1f}%)")

    return obss
######

def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer, train_type='clean', noise_rate=0.0 , model=None):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0
    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
        print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                obss += [states]
                actions += [ac[0]]
                stepwise_returns += [ret[0]]
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]
                i += 1
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            transitions_per_buffer[buffer_num] = i
        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

    # ここでノイズを適用
    obss = apply_noise(obss, train_type, noise_rate, model)
    ######

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return obss, actions, returns, done_idxs, rtg, timesteps
