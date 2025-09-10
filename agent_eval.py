"""
MicroRTS PPO 에이전트 평가 스크립트

이 스크립트는 훈련된 PPO 에이전트를 다양한 AI와 대전하여 성능을 평가합니다.

주요 기능:
1. 다양한 AI 에이전트와의 대전 평가 (13종류의 AI)
2. GridNet과 일반 아키텍처 지원
3. 게임 결과 시각화 및 로깅
4. 비디오 녹화 및 wandb 업로드
5. 상세한 통계 수집 및 분석

사용법:
    python agent_eval.py --agent-model-path path/to/model.pt --num-eval-runs 10

참고 논문: http://proceedings.mlr.press/v97/han19a/han19a.pdf

작성자: 20250909 dasom
"""
# 가상 디스플레이 설정 (헤드리스 환경에서 비디오 녹화를 위해)
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

# PyTorch 관련 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# 기타 유틸리티 라이브러리
import argparse
from distutils.util import strtobool
import numpy as np
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv, MicroRTSVecEnv
from gym_microrts import microrts_ai
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
import matplotlib.pyplot as plt
import pandas as pd
import glob

if __name__ == "__main__":
    # 명령행 인자 파싱 설정
    parser = argparse.ArgumentParser(description='PPO agent')
    
    # 공통 인자들
    parser.add_argument('--exp-name', type=str, default="ppo_gridnet_diverse_encode_decode",
                        help='실험 이름 (에이전트 아키텍처 타입을 결정)')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='에이전트 성능 비디오 캡처 여부')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="wandb 프로젝트 이름")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # 알고리즘 특화 인자들
    parser.add_argument('--num-bot-envs', type=int, default=0,
                        help='봇 게임 환경 수 (16개 봇 환경 = 16개 게임)')
    parser.add_argument('--num-selfplay-envs', type=int, default=16,
                        help='셀프플레이 환경 수 (16개 셀프플레이 환경 = 8개 게임)')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='게임 환경당 스텝 수')
    parser.add_argument('--num-eval-runs', type=int, default=10,
                        help='각 AI와의 평가 게임 수')
    parser.add_argument('--agent-model-path', type=str, default="trained_models/ppo_gridnet_diverse_encode_decode/agent-1.pt",
                        help="에이전트 모델 파일 경로")
    parser.add_argument('--max-steps', type=int, default=2000,
                        help="MicroRTS에서 최대 게임 스텝 수")

    # 인자 파싱 및 기본값 설정
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs

# 에피소드 모니터링을 위한 벡터화된 환경 래퍼
class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None  # 에피소드별 누적 보상
        self.eplens = None  # 에피소드별 길이
        self.epcount = 0    # 총 에피소드 수
        self.tstart = time.time()  # 시작 시간

    def reset(self):
        """환경 리셋 및 에피소드 통계 초기화"""
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')  # 부동소수점 배열
        self.eplens = np.zeros(self.num_envs, 'i')  # 정수 배열
        return obs

    def step_wait(self):
        """스텝 실행 및 에피소드 완료 시 통계 기록"""
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews  # 누적 보상 업데이트
        self.eplens += 1     # 에피소드 길이 증가

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:  # 에피소드가 완료된 경우
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                # 에피소드 정보 생성 (보상, 길이, 시간)
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                info['episode'] = epinfo
                self.epcount += 1
                # 완료된 에피소드 통계 리셋
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos

# MicroRTS 통계 기록을 위한 벡터화된 환경 래퍼
class MicroRTSStatsRecorder(VecEnvWrapper):

    def reset(self):
        """환경 리셋 및 원시 보상 기록 초기화"""
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]  # 환경별 원시 보상 리스트
        return obs

    def step_wait(self):
        """스텝 실행 및 원시 보상 수집, 에피소드 완료 시 통계 계산"""
        obs, rews, dones, infos = self.venv.step_wait()
        
        # 각 환경의 원시 보상 수집
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]] 
        
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:  # 에피소드가 완료된 경우
                info = infos[i].copy()
                # 에피소드 동안의 모든 원시 보상 합계 계산
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]  # 보상 함수 이름들
                # MicroRTS 통계를 딕셔너리로 저장
                info['microrts_stats'] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[i] = []  # 완료된 에피소드의 원시 보상 리셋
                newinfos[i] = info
        return obs, rews, dones, newinfos

# 환경 설정 및 로깅 초기화
experiment_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
# 하이퍼파라미터를 텍스트로 기록
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

# 프로덕션 모드에서 wandb 초기화
if args.prod_mode:
    import wandb
    run = wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, 
                     sync_tensorboard=True, config=vars(args), name=experiment_name, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# 재현 가능한 결과를 위한 시드 설정
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
# 평가할 AI 에이전트들 정의 (다양한 난이도와 전략)
all_ais = {
    "randomBiasedAI": microrts_ai.randomBiasedAI,      # 편향된 랜덤 AI
    "randomAI": microrts_ai.randomAI,                  # 순수 랜덤 AI
    "passiveAI": microrts_ai.passiveAI,                # 수동적 AI
    "workerRushAI": microrts_ai.workerRushAI,          # 워커 러시 전략 AI
    "lightRushAI": microrts_ai.lightRushAI,            # 라이트 러시 전략 AI
    "coacAI": microrts_ai.coacAI,                      # CoacAI (강력한 AI)
    "naiveMCTSAI": microrts_ai.naiveMCTSAI,            # 나이브 MCTS AI
    "mixedBot": microrts_ai.mixedBot,                  # 혼합 전략 봇
    "rojo": microrts_ai.rojo,                          # Rojo AI
    "izanagi": microrts_ai.izanagi,                    # Izanagi AI
    "tiamat": microrts_ai.tiamat,                      # Tiamat AI
    "droplet": microrts_ai.droplet,                    # Droplet AI
    "guidedRojoA3N": microrts_ai.guidedRojoA3N         # 가이드된 Rojo A3N AI
}

# AI 이름과 인스턴스 분리
ai_names, ais = list(all_ais.keys()), list(all_ais.values())
# 각 AI별 매치 통계 초기화 (패배, 무승부, 승리)
ai_match_stats = dict(zip(ai_names, np.zeros((len(ais), 3))))
args.num_envs = len(ais)
ai_envs = []

# GridNet 아키텍처를 사용하는 실험들
gridnet_exps = ["ppo_gridnet_diverse_impala", "ppo_gridnet_coacai", "ppo_gridnet_naive", 
                "ppo_gridnet_diverse", "ppo_gridnet_diverse_encode_decode", 
                "ppo_gridnet_coacai_naive", "ppo_gridnet_coacai_partial_mask",
                "ppo_gridnet_coacai_no_mask", "ppo_gridnet_selfplay_encode_decode", 
                "ppo_gridnet_selfplay_diverse_encode_decode"]
# 각 AI에 대해 환경 생성
for i in range(len(ais)):
    if args.exp_name in gridnet_exps:
        # GridNet 아키텍처용 환경 (그리드 기반 액션)
        envs = MicroRTSGridModeVecEnv(
            num_bot_envs=1,                    # 봇 환경 수
            num_selfplay_envs=0,               # 셀프플레이 환경 수
            max_steps=args.max_steps,          # 최대 스텝 수
            render_theme=2,                    # 렌더링 테마
            ai2s=[ais[i]],                     # 상대 AI
            map_path="maps/16x16/basesWorkers16x16A.xml",  # 맵 파일
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])  # 보상 가중치
        )
        # 환경 래퍼 추가
        envs = MicroRTSStatsRecorder(envs)     # MicroRTS 통계 기록
        envs = VecMonitor(envs)                # 에피소드 모니터링
        envs = VecVideoRecorder(envs, f'videos/{experiment_name}/{ai_names[i]}',
                                record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)
    else:
        # 일반 아키텍처용 환경 (단위 기반 액션)
        envs = MicroRTSVecEnv(
            num_envs=1,                        # 환경 수
            max_steps=args.max_steps,          # 최대 스텝 수
            render_theme=2,                    # 렌더링 테마
            ai2s=[ais[i]],                     # 상대 AI
            map_path="maps/16x16/basesWorkers16x16.xml",  # 맵 파일
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])  # 보상 가중치
        )
        # 환경 래퍼 추가
        envs = MicroRTSStatsRecorder(envs)     # MicroRTS 통계 기록
        envs = VecMonitor(envs)                # 에피소드 모니터링
        envs = VecVideoRecorder(envs, f'videos/{experiment_name}/{ai_names[i]}',
                                record_video_trigger=lambda x: x % 4000 == 0, video_length=2000)
    ai_envs += [envs]

# 액션 공간 검증
assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

# ==================== 에이전트 아키텍처 정의 ====================

# 마스크된 카테고리컬 분포 클래스 (유효하지 않은 액션을 마스킹)
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            # 마스크가 없는 경우 일반 카테고리컬 분포 사용
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            # 마스크가 있는 경우 유효하지 않은 액션의 로짓을 매우 작은 값으로 설정
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        """마스크를 고려한 엔트로피 계산"""
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        # 마스크된 액션은 엔트로피 계산에서 제외
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

# 스케일링 레이어 (값에 상수를 곱함)
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# 텐서 차원 순서 변경 레이어
class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)

# 레이어 초기화 함수 (직교 초기화 사용)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# 잔차 블록 (ResNet 스타일)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x  # 잔차 연결을 위한 입력 저장
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs  # 잔차 연결

# 컨볼루션 시퀀스 (Conv + MaxPool + Residual Blocks)
class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        # 초기 컨볼루션 레이어
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, 
                              kernel_size=3, padding=1)
        # 두 개의 잔차 블록
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        # 맥스 풀링으로 공간 크기 절반으로 축소
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        """출력 텐서의 크기 계산"""
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

# ==================== 에이전트 아키텍처별 정의 ====================

if args.exp_name == "ppo_gridnet_diverse_impala":
    # GridNet + IMPALA 아키텍처 에이전트 (다양한 AI와 대전)
    class Agent(nn.Module):
        def __init__(self, mapsize=16*16):
            super(Agent, self).__init__()
            self.mapsize = mapsize  # 맵 크기 (16x16 = 256)
            h, w, c = envs.observation_space.shape
            shape = (c, h, w)  # (채널, 높이, 너비)
            
            # 컨볼루션 시퀀스들 구성
            conv_seqs = []
            for out_channels in [16, 32, 32]:  # 점진적으로 채널 수 증가
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            
            # 완전연결층 추가
            conv_seqs += [
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
                nn.ReLU(),
            ]
            self.network = nn.Sequential(*conv_seqs)
            
            # 액터 헤드 (각 그리드 셀에 대한 액션 예측)
            self.actor = layer_init(nn.Linear(256, self.mapsize*envs.action_space.nvec[1:].sum()), std=0.01)
            # 크리틱 헤드 (상태 가치 예측)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)
            
            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                action = action.view(-1,action.shape[-1]).T
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum()+1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name == "ppo_diverse_impala":
    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            h, w, c = envs.observation_space.shape
            shape = (c, h, w)
            conv_seqs = []
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
                nn.ReLU(),
            ]
            self.network = nn.Sequential(*conv_seqs)
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)
            
            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name == "ppo_coacai":
    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32*6*6, 256)),
                nn.ReLU(),)
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)
            
            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name == "ppo_coacai_no_mask":
    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32*6*6, 256)),
                nn.ReLU(),)
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)
            
            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                
                # this removes the unit action mask
                source_unit_action_mask[:] = 1
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name == "ppo_gridnet_naive":
    class Agent(nn.Module):
        def __init__(self, mapsize=16*16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32*6*6, 256)),
                nn.ReLU(),)
            self.actor = layer_init(nn.Linear(256, self.mapsize*envs.action_space.nvec[1:].sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)
            
            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                action = action.view(-1,action.shape[-1]).T
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [Categorical(logits=logits) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum()+1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name in ["ppo_gridnet_diverse", "ppo_gridnet_coacai", "ppo_gridnet_coacai_naive"]:
    class Agent(nn.Module):
        def __init__(self, mapsize=16*16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32*6*6, 256)),
                nn.ReLU(),)
            self.actor = layer_init(nn.Linear(256, self.mapsize*envs.action_space.nvec[1:].sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)
            
            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                action = action.view(-1,action.shape[-1]).T
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum()+1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name == "ppo_diverse":
    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32*6*6, 256)),
                nn.ReLU(),)
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)
            
            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name in ["ppo_gridnet_diverse_encode_decode", "ppo_gridnet_selfplay_diverse_encode_decode", "ppo_gridnet_selfplay_encode_decode"]:
    class Transpose(nn.Module):
        def __init__(self, permutation):
            super().__init__()
            self.permutation = permutation
    
        def forward(self, x):
            return x.permute(self.permutation)
    class Encoder(nn.Module):
        def __init__(self, input_channels):
            super().__init__()
            self._encoder = nn.Sequential(
                Transpose((0, 3, 1, 2)),
                layer_init(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.ReLU(),
                layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
                nn.MaxPool2d(3, stride=2, padding=1),
            )
    
        def forward(self, x):
            return self._encoder(x)
    
    
    class Decoder(nn.Module):
        def __init__(self, output_channels):
            super().__init__()
    
            self.deconv = nn.Sequential(
                layer_init(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1)),
                Transpose((0, 2, 3, 1)),
            )
    
        def forward(self, x):
            return self.deconv(x)
    
    
    class Agent(nn.Module):
        def __init__(self, mapsize=16 * 16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            h, w, c = envs.observation_space.shape
    
            self.encoder = Encoder(c)
    
            self.actor = Decoder(78)
    
            self.critic = nn.Sequential(
                nn.Flatten(),
                layer_init(nn.Linear(256, 128), std=1),
                nn.ReLU(),
                layer_init(nn.Linear(128, 1), std=1),
            )
    
        def forward(self, x):
            return self.encoder(x)  # "bhwc" -> "bchw"
    
        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.reshape(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)
    
            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                         dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                      zip(split_logits, split_invalid_action_masks)]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
                action = action.view(-1, action.shape[-1]).T
                split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                         dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                      zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks
    
        def get_value(self, x):
            return self.critic(self.forward(x))

elif args.exp_name == "ppo_coacai_naive":
    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32*6*6, 256)),
                nn.ReLU(),)
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)
            
            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                
                # remove the mask on action parameters, which is a similar setup to pysc2
                source_unit_action_mask[:,6:] = 1
                
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name == "ppo_coacai_partial_mask":
    class Agent(nn.Module):
        def __init__(self, frames=4):
            super(Agent, self).__init__()
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32*6*6, 256)),
                nn.ReLU(),)
            self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)
            
            if action is None:
                # 1. select source unit based on source unit mask
                source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(1, -1))
                multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
                action_components = [multi_categoricals[0].sample()]
                # 2. select action type and parameter section based on the
                #    source-unit mask of action type and parameters
                # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                source_unit_action_mask = torch.Tensor(
                    np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(1, -1))
                split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
                multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)]
                invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
                action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
                action = torch.stack(action_components)
            else:
                split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
                multi_categoricals = [Categorical(logits=logits) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name == "ppo_gridnet_coacai_partial_mask":
    class Agent(nn.Module):
        def __init__(self, mapsize=16*16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32*6*6, 256)),
                nn.ReLU(),)
            self.actor = layer_init(nn.Linear(256, self.mapsize*envs.action_space.nvec[1:].sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)
            
            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                
                # remove the mask on action parameters, which is a similar setup to pysc2
                invalid_action_masks[:,6:] = 1
                
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                action = action.view(-1,action.shape[-1]).T
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum()+1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
elif args.exp_name == "ppo_gridnet_coacai_no_mask":
    class Agent(nn.Module):
        def __init__(self, mapsize=16*16):
            super(Agent, self).__init__()
            self.mapsize = mapsize
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=2)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32*6*6, 256)),
                nn.ReLU(),)
            self.actor = layer_init(nn.Linear(256, self.mapsize*envs.action_space.nvec[1:].sum()), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

        def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
            logits = self.actor(self.forward(x))
            grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
            split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)
            
            if action is None:
                invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                real_invalid_action_masks = invalid_action_masks.clone()

                # remove masks
                invalid_action_masks[:] = 1
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            else:
                invalid_action_masks = invalid_action_masks.view(-1,invalid_action_masks.shape[-1])
                real_invalid_action_masks = invalid_action_masks.clone()
                
                # remove masks
                invalid_action_masks[:] = 1
                action = action.view(-1,action.shape[-1]).T
                split_invalid_action_masks = torch.split(invalid_action_masks[:,1:], envs.action_space.nvec[1:].tolist(), dim=1)
                multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            num_predicted_parameters = len(envs.action_space.nvec) - 1
            logprob = logprob.T.view(-1, 256, num_predicted_parameters)
            entropy = entropy.T.view(-1, 256, num_predicted_parameters)
            action = action.T.view(-1, 256, num_predicted_parameters)
            invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum()+1)
            real_invalid_action_masks = real_invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum()+1)
            return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

        def get_value(self, x):
            return self.critic(self.forward(x))
else:
    raise Exception("incorrect agent selected")

# ==================== 에이전트 초기화 및 로딩 ====================

# 에이전트 생성 및 디바이스로 이동
agent = Agent().to(device)
# 훈련된 모델 가중치 로드
agent.load_state_dict(torch.load(args.agent_model_path, map_location=device))
agent.eval()  # 평가 모드로 설정

# 모델 정보 출력
print("Model's state_dict:")
for param_tensor in agent.state_dict():
    print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
total_params = sum([param.nelement() for param in agent.parameters()])
print("Model's total parameters:", total_params)
writer.add_scalar(f"charts/total_parameters", total_params, 0)

# ==================== 평가 데이터 저장소 초기화 ====================
mapsize = 16*16
action_space_shape = (mapsize, envs.action_space.shape[0] - 1)
invalid_action_shape = (mapsize, envs.action_space.nvec[1:].sum()+1)

# 평가 중 데이터 저장을 위한 텐서들 (실제로는 사용되지 않음)
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)
# ==================== 평가 루프 시작 ====================
global_step = 0
start_time = time.time()

# 각 AI와의 대전 평가
for envs_idx, envs in enumerate(ai_envs):
    next_obs = torch.Tensor(envs.reset()).to(device)  # 환경 초기화
    next_done = torch.zeros(args.num_envs).to(device)  # 완료 상태 초기화
    game_count = 0  # 게임 수 카운터
    from jpype.types import JArray, JInt  # Java 배열 타입 (MicroRTS용)
    
    while True:
        # envs.render()  # 렌더링 (주석 처리됨)
        
        # ==================== 액션 선택 ====================
        with torch.no_grad():  # 그래디언트 계산 비활성화 (평가 모드)
            action, logproba, _, invalid_action_mask = agent.get_action(next_obs, envs=envs)

        # ==================== 액션 실행 및 환경 스텝 ====================
        if args.exp_name in gridnet_exps:
            # GridNet 아키텍처: 소스 유닛 인덱스를 액션에 추가
            real_action = torch.cat([
                torch.stack(
                    [torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)
            ]).unsqueeze(2), action], 2)
            
            # real_action의 형태: (num_envs, map_height*map_width, 8)
            # 맵의 각 셀에 대해 액션을 예측하지만, 소스 유닛이 없는 셀의 액션은 무효
            # 성능 향상을 위해 유효한 액션만 추출
            real_action = real_action.cpu().numpy()
            valid_actions = real_action[invalid_action_mask[:,:,0].bool().cpu().numpy()]
            valid_actions_counts = invalid_action_mask[:,:,0].sum(1).long().cpu().numpy()
            
            # Java 배열 형태로 변환 (MicroRTS가 요구하는 형식)
            java_valid_actions = []
            valid_action_idx = 0
            for env_idx, valid_action_count in enumerate(valid_actions_counts):
                java_valid_action = []
                for c in range(valid_action_count):
                    java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
                    valid_action_idx += 1
                java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
            java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)
        else:
            # 일반 아키텍처: 단위 기반 액션
            java_valid_actions = action.T.cpu().numpy()
        
        # ==================== 환경 스텝 실행 ====================
        try:
            next_obs, rs, ds, infos = envs.step(java_valid_actions)
            next_obs = torch.Tensor(next_obs).to(device)
        except Exception as e:
            e.printStackTrace()
            raise

        # ==================== 에피소드 완료 처리 ====================
        for idx, info in enumerate(infos):
            if 'episode' in info.keys():  # 에피소드가 완료된 경우
                # print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                # writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                
                # 게임 결과 출력 및 통계 업데이트
                print("against", ai_names[envs_idx], info['microrts_stats']['WinLossRewardFunction'])
                if info['microrts_stats']['WinLossRewardFunction'] == -1.0:
                    ai_match_stats[ai_names[envs_idx]][0] += 1  # 패배
                elif info['microrts_stats']['WinLossRewardFunction'] == 0.0:
                    ai_match_stats[ai_names[envs_idx]][1] += 1  # 무승부
                elif info['microrts_stats']['WinLossRewardFunction'] == 1.0:
                    ai_match_stats[ai_names[envs_idx]][2] += 1  # 승리
                game_count += 1
                
                # writer.add_scalar(f"charts/episode_reward/{key}", , global_step)
                # for key in info['microrts_stats']:
                #     writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
                # print("=============================================")
                # break
        
        # ==================== 평가 완료 체크 ====================
        if game_count >= args.num_eval_runs:  # 설정된 게임 수만큼 완료
            envs.close()
            # TensorBoard에 결과 기록
            for (label, val) in zip(["loss", "tie", "win"], ai_match_stats[ai_names[envs_idx]]):
                writer.add_scalar(f"charts/{ai_names[envs_idx]}/{label}", val, 0)
            
            # 프로덕션 모드에서 비디오 업로드
            if args.prod_mode and args.capture_video:
                video_files = glob.glob(f'videos/{experiment_name}/{ai_names[envs_idx]}/*.mp4')
                for video_file in video_files:
                    print(video_file)
                    wandb.log({f"RL agent against {ai_names[envs_idx]}": wandb.Video(video_file)})
                # labels, values = ["loss", "tie", "win"], ai_match_stats[ai_names[envs_idx]]
                # data = [[label, val] for (label, val) in zip(labels, values)]
                # table = wandb.Table(data=data, columns = ["match result", "number of games"])
                # wandb.log({ai_names[envs_idx]: wandb.plot.bar(table, "match result", "number of games", title=f"RL agent against {ai_names[envs_idx]}")})
            break

# ==================== 결과 시각화 및 저장 ====================

# 매치 결과 시각화 (막대 그래프)
n_rows, n_cols = 3, 5
fig = plt.figure(figsize=(5*3, 4*3))
for i, var_name in enumerate(ai_names):
    ax = fig.add_subplot(n_rows, n_cols, i+1)
    ax.bar(["loss", "tie", "win"], ai_match_stats[var_name])
    ax.set_title(var_name)
fig.suptitle(args.agent_model_path)
fig.tight_layout()

# 누적 매치 결과 계산
cumulative_match_results = np.array(list(ai_match_stats.values())).sum(0)
cumulative_match_results_rate = cumulative_match_results / cumulative_match_results.sum()

# 프로덕션 모드에서 결과 로깅
if args.prod_mode:
    wandb.log({"Match results": wandb.Image(fig)})
    for (label, val) in zip(["loss", "tie", "win"], cumulative_match_results):
        writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
    for (label, val) in zip(["loss rate", "tie rate", "win rate"], cumulative_match_results_rate):
        writer.add_scalar(f"charts/cumulative_match_results/{label}", val, 0)
    # labels, values = ["loss", "tie", "win"], cumulative_match_results
    # data = [[label, val] for (label, val) in zip(labels, values)]
    # table = wandb.Table(data=data, columns = ["cumulative match result", "number of games"])
    # wandb.log({"cumulative": wandb.plot.bar(table, "cumulative match result", "number of games", title="RL agent cumulative results")})

# ==================== 정리 ====================
envs.close()
writer.close()
