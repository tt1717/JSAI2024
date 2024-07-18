"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image
import imageio
import wandb
#import time
import skimage as sk


class TrainerConfig:
    # optimization parameters
    max_epochs = 1
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    ##########################################
    eval_type = 'clean'  # デフォルト値を'clean'に設定
    ##########################################

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, log_info):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.log_info = log_info

        # total_it
        self.total_it = 0

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)


    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        logs = dict()

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # total_it 
                self.total_it += 1
                ##########################################
                
                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

                    # total iters to wandb
                    if self.log_info['log_to_wandb']:
                        wandb.log({"total iter": self.total_it})

                    # train loss and lr to wandb
                    if self.log_info['log_to_wandb']:
                        wandb.log({"train loss": loss.item(), "total iter": self.total_it})
                        wandb.log({"train lr": lr, "total iter": self.total_it})
                    ##########################################
                        
            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')
        # best_return = -float('inf')

        # T_rewards_list
        T_rewards_list = []
        ##########################################

        # eval_returns and stds
        eval_returns, eval_stds = [], []

        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

            # -- pass in target returns
            if self.config.model_type == 'bc':
                eval_return, eval_std, T_rewards = self.get_returns(0, epoch)
            elif self.config.model_type == 'dt':
                if self.config.game == 'Breakout':
                    eval_return, eval_std, T_rewards = self.get_returns(90, epoch)
                elif self.config.game == 'Seaquest':
                    eval_return, eval_std, T_rewards = self.get_returns(1150, epoch)
                elif self.config.game == 'Qbert':
                    eval_return, eval_std, T_rewards = self.get_returns(14000, epoch)
                elif self.config.game == 'Pong':
                    eval_return, eval_std, T_rewards = self.get_returns(20, epoch)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
            
            # eval_returns and eval_stds and T_rewards_list
            eval_returns.append(eval_return)
            eval_stds.append(eval_std)
            T_rewards_list.append(T_rewards)
            ##########################################

        return eval_returns, eval_stds, T_rewards_list


    def get_returns(self, ret, epoch):
        self.model.train(False)
        args=Args(self.config.game.lower(), self.config.seed, self.config.eval_type.lower())
        env = Env(args)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True

        # 評価テストの回数を10から5に変更
        for i in range(5):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model, state, 1, temperature=1.0, sample=True, actions=None,
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))

                env.render() # add render
                      
        env.save_gif(self.config.game, self.config.seed, self.config.model_type, epoch)
        env.close()

        eval_return = sum(T_rewards) / len(T_rewards)
        eval_std = np.std(T_rewards)
        print("target return: %d, eval return: %d, eval std: %d" % (ret, eval_return, eval_std))
        ##########################################
    
        # 最後のエポックでのみGIF動画をW&Bにアップロード
        if epoch == self.config.max_epochs - 1:
            gif_filename = 'result/{}_{}_{}_{}.gif'.format(self.config.game, self.config.model_type, self.config.seed, epoch)
            if self.log_info['log_to_wandb']:
                wandb.log({"eval gif": wandb.Video(gif_filename, fps=4, format="gif")})
        ##########################################

        self.model.train(True)

        # eval_average_return and std to wandb
        if self.log_info['log_to_wandb']:
            wandb.log({"eval average return": eval_return, "total iter": self.total_it})
            wandb.log({"eval std": np.std(T_rewards), "total iter": self.total_it})
        ##########################################

        return eval_return, eval_std, T_rewards

############################################
# ガウシアンノイズ関数を追加
def gaussian_noise(tensor, severity):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    # テンソルをNumPy配列に変換
    np_tensor = tensor.cpu().numpy()
    # NumPy配列にガウシアンノイズを適用
    noisy_np_tensor = np.clip(np_tensor + np.random.normal(scale=c, size=np_tensor.shape), 0, 1)
    # NumPy配列をテンソルに戻す
    noisy_tensor = torch.from_numpy(noisy_np_tensor).to(tensor.device)
    return noisy_tensor

# インパルスノイズ関数を追加
def impulse_noise(tensor, severity):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    # テンソルをNumPy配列に変換
    np_tensor = tensor.cpu().numpy()
    # NumPy配列にノイズを適用
    noisy_np_tensor = sk.util.random_noise(np_tensor, mode='s&p', amount=c)
    # NumPy配列をテンソルに戻す
    noisy_tensor = torch.from_numpy(noisy_np_tensor).to(tensor.device)
    return noisy_tensor

# ショットノイズ関数を追加
def shot_noise(tensor, severity):
    c = [60, 25, 12, 5, 3][severity - 1]
    # テンソルをNumPy配列に変換
    np_tensor = tensor.cpu().numpy()
    # NumPy配列にショットノイズを適用
    noisy_np_tensor = np.clip(np.random.poisson(np_tensor * c) / float(c), 0, 1)
    # NumPy配列をテンソルに戻す
    noisy_tensor = torch.from_numpy(noisy_np_tensor).to(tensor.device)
    return noisy_tensor

# スペックルノイズ関数を追加
def speckle_noise(tensor, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    # テンソルをNumPy配列に変換
    np_tensor = tensor.cpu().numpy()
    # NumPy配列にスペックルノイズを適用
    noisy_np_tensor = np.clip(np_tensor + np_tensor * np.random.normal(size=np_tensor.shape, scale=c), 0, 1)
    # NumPy配列をテンソルに戻す
    noisy_tensor = torch.from_numpy(noisy_np_tensor).to(tensor.device)
    return noisy_tensor
############################################


class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode
        self.frames = []
        ############################################
        self.args = args
        ############################################

    ############################################
    def apply_noise(self, observation):
        if self.args.eval_type == 'gaus':
            return gaussian_noise(observation, severity=1)
        elif self.args.eval_type == 'shot':
            return shot_noise(observation, severity=1)
        elif self.args.eval_type == 'imp':
            return impulse_noise(observation, severity=1)
        elif self.args.eval_type == 'spe':
            return speckle_noise(observation, severity=1)
        return observation  # 'clean'または該当しない場合、そのまま返す
    ############################################
    

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()

        ############################################
        observation_noisy = self.apply_noise(observation)
        self.state_buffer.append(observation_noisy)
        ############################################

        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]

        
        ############################################
        observation_noisy = self.apply_noise(observation)
        self.state_buffer.append(observation_noisy)
        ############################################

        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        
        ############################################
        # 元のRGB画像を取得し、NumPy配列のコピーを作成
        frame_rgb = self.ale.getScreenRGB()[:, :, ::-1].copy()  # OpenCV用にBGRに変換し、コピーを作成

        # 画像をtorchテンソルに変換
        frame_tensor = torch.from_numpy(frame_rgb).float().div(255).to(self.device)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # チャネル次元を調整

        # ノイズを適用
        frame_tensor_noisy = self.apply_noise(frame_tensor)  # apply_noiseを利用してノイズを適用

        # テンソルをNumPy配列に戻す
        frame_noisy_np = frame_tensor_noisy.squeeze().permute(1, 2, 0).cpu().numpy()
        frame_noisy_np = (frame_noisy_np * 255).astype(np.uint8)

        # ノイズ付きの画像を保存し、表示
        self.frames.append(cv2.cvtColor(frame_noisy_np, cv2.COLOR_BGR2RGB))
        cv2.imshow('screen', frame_noisy_np)
        cv2.waitKey(1)
        ############################################

    def save_gif(self, game_name, seed, mode_type, epoch):
        # save gif with pillow
        frames = []
        for frame in self.frames:
            frames.append(Image.fromarray(frame))
        frames[0].save('result/{}_{}_{}_{}.gif'.format(game_name, mode_type, seed, epoch), save_all=True, append_images=frames[1:], duration=30, loop=0)

    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed, eval_type):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
        self.eval_type = eval_type  # eval_typeを追加

