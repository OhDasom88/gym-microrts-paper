"""
PPO CoacAI No Mask 에이전트 훈련 스크립트

이 스크립트는 CoacAI와 대전하는 PPO 에이전트를 훈련합니다.
액션 마스킹을 사용하지 않는 버전입니다.

주요 특징:
- CoacAI와의 대전
- 액션 마스킹 없음
- 단위 기반 액션 공간

참고 논문: https://arxiv.org/abs/1707.06347
"""

# ==================== 라이브러리 임포트 ====================

# PyTorch 관련
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# 일반 라이브러리
import argparse
from distutils.util import strtobool
import numpy as np
import gym
import gym_microrts
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from gym.wrappers import TimeLimit, Monitor

from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder

if __name__ == "__main__":
    # ==================== 인수 파싱 ====================
    
    parser = argparse.ArgumentParser(description='PPO agent')
    
    # ==================== 공통 인수 ====================
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='실험 이름')
    parser.add_argument('--gym-id', type=str, default="MicrortsDefeatCoacAIShaped-v3",
                        help='Gym 환경 ID')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='옵티마이저 학습률')
    parser.add_argument('--seed', type=int, default=1,
                        help='실험 시드')
    parser.add_argument('--total-timesteps', type=int, default=100000000,
                        help='총 훈련 타임스텝')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='PyTorch 결정적 연산 사용 여부')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='CUDA 사용 여부')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='프로덕션 모드 실행 및 wandb 로깅 사용')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='에이전트 성능 비디오 캡처 여부')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="wandb 프로젝트 이름")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="wandb 엔티티 (팀) 이름")

    # ==================== PPO 알고리즘 특화 인수 ====================
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='미니배치 수')
    parser.add_argument('--num-envs', type=int, default=24,
                        help='병렬 게임 환경 수')
    parser.add_argument('--num-steps', type=int, default=512,
                        help='게임 환경당 스텝 수')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='할인 인자 감마')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='일반화 어드밴티지 추정을 위한 람다')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="엔트로피 계수")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="가치 함수 계수")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='그래디언트 클리핑을 위한 최대 노름')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="서로게이트 클리핑 계수")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="정책 업데이트를 위한 K 에포크")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='KL 발산 기반 조기 정지 사용 여부')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='KL 발산이 목표값을 초과할 때 이전 정책으로 롤백 사용 여부')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='KL 발산 목표값')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='어드밴티지 계산을 위한 GAE 사용 여부')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="어드밴티지 정규화 사용 여부")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="정책 및 가치 네트워크의 학습률 어닐링 사용 여부")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='논문에 따른 가치 함수 클리핑 손실 사용 여부')

    # 인수 파싱 및 시드 설정
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# ==================== 환경 및 배치 크기 계산 ====================

# 배치 크기 계산
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)

# ==================== 환경 래퍼 클래스들 ====================

class VecMonitor(VecEnvWrapper):
    """
    에피소드 보상과 길이를 모니터링하는 VecEnv 래퍼
    
    각 환경의 에피소드별 보상과 길이를 추적하고,
    에피소드가 종료될 때 통계 정보를 제공합니다.
    """
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None  # 에피소드별 누적 보상
        self.eplens = None  # 에피소드별 길이
        self.epcount = 0    # 총 에피소드 수
        self.tstart = time.time()  # 시작 시간

    def reset(self):
        """환경 리셋 및 에피소드 통계 초기화"""
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')  # 누적 보상 초기화
        self.eplens = np.zeros(self.num_envs, 'i')  # 에피소드 길이 초기화
        return obs

    def step_wait(self):
        """스텝 실행 및 에피소드 통계 업데이트"""
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews  # 누적 보상 업데이트
        self.eplens += 1     # 에피소드 길이 증가

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:  # 에피소드 종료 시
                info = infos[i].copy()
                ret = self.eprets[i]  # 최종 누적 보상
                eplen = self.eplens[i]  # 에피소드 길이
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                info['episode'] = epinfo  # 에피소드 정보 추가
                self.epcount += 1
                self.eprets[i] = 0  # 보상 초기화
                self.eplens[i] = 0  # 길이 초기화
                newinfos[i] = info
        return obs, rews, dones, newinfos

class VecPyTorch(VecEnvWrapper):
    """
    관찰과 보상을 PyTorch 텐서로 변환하는 VecEnv 래퍼
    """
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        """환경 리셋 및 관찰을 PyTorch 텐서로 변환"""
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        """비동기 액션 실행"""
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        """스텝 대기 및 관찰을 PyTorch 텐서로 변환"""
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

class MicroRTSStatsRecorder(VecEnvWrapper):
    """
    MicroRTS 특화 통계를 기록하는 VecEnv 래퍼
    
    원시 보상(raw rewards)과 승/패 정보를 추적하여
    MicroRTS 환경의 상세한 통계를 제공합니다.
    """
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        """환경 리셋 및 원시 보상 기록 초기화"""
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]  # 원시 보상 기록
        return obs

    def step_wait(self):
        """스텝 실행 및 원시 보상 기록"""
        obs, rews, dones, infos = self.venv.step_wait()
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]  # 원시 보상 누적
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:  # 에피소드 종료 시
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)  # 원시 보상 합계
                raw_names = [str(rf) for rf in self.rfs]  # 보상 함수 이름들
                info['microrts_stats'] = dict(zip(raw_names, raw_rewards))  # 통계 정보 추가
                self.raw_rewards[i] = []  # 원시 보상 초기화
                newinfos[i] = info
        return obs, rews, dones, newinfos

# ==================== 환경 설정 ====================

# 실험 이름 생성 (gym_id + exp_name + seed + timestamp)
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

# TensorBoard 로거 설정
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
# Weights & Biases 로깅 설정
if args.prod_mode:
    import wandb
    run = wandb.init(
        project=args.wandb_project_name, entity=args.wandb_entity,
        # sync_tensorboard=True,
        config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 50

# ==================== 시드 설정 및 환경 생성 ====================

# 디바이스 설정 (CUDA 사용 가능 시 GPU, 아니면 CPU)
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

# 재현 가능한 결과를 위한 시드 설정
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

# MicroRTS 벡터 환경 생성
envs = MicroRTSVecEnv(
    num_envs=args.num_envs,  # 환경 수
    max_steps=2000,          # 최대 스텝 수
    render_theme=2,          # 렌더링 테마
    ai2s=[microrts_ai.coacAI for _ in range(args.num_envs)],  # CoacAI 상대
    map_path="maps/16x16/basesWorkers16x16.xml",              # 맵 경로
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])  # 보상 가중치
)

# 환경 래퍼 적용
envs = MicroRTSStatsRecorder(envs, args.gamma)  # MicroRTS 통계 기록
envs = VecMonitor(envs)                         # 에피소드 모니터링
envs = VecPyTorch(envs, device)                # PyTorch 텐서 변환
if args.capture_video:
    envs = VecVideoRecorder(envs, f'videos/{experiment_name}',
                            record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)  # 비디오 녹화
assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

# ==================== 에이전트 아키텍처 ====================

class CategoricalMasked(Categorical):
    """
    액션 마스킹을 지원하는 Categorical 분포
    
    유효하지 않은 액션의 로짓을 매우 작은 값으로 설정하여
    선택되지 않도록 합니다.
    """
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            # 마스크된 액션의 로짓을 매우 작은 값으로 설정
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        """마스킹을 고려한 엔트로피 계산"""
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

class Scale(nn.Module):
    """입력을 스케일링하는 간단한 레이어"""
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """레이어 가중치와 편향을 초기화하는 함수"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """
    단위 기반 액션 공간을 사용하는 PPO 에이전트
    
    컨볼루션 네트워크를 사용하여 관찰을 처리하고,
    액터-크리틱 구조를 사용합니다.
    """
    def __init__(self, frames=4):
        super(Agent, self).__init__()
        # 특징 추출 네트워크
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),  # 첫 번째 컨볼루션
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),            # 두 번째 컨볼루션
            nn.ReLU(),
            nn.Flatten(),                                            # 평면화
            layer_init(nn.Linear(32*6*6, 256)),                     # 완전 연결 레이어
            nn.ReLU(),)
        
        # 액터와 크리틱 헤드
        self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        """순전파: 관찰을 네트워크에 통과시켜 특징 추출"""
        return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

    def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
        """액션 선택 및 로그 확률 계산"""
        logits = self.actor(self.forward(x))  # 액터 네트워크로 로짓 계산
        split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)  # 액션 타입별로 분할
        
        if action is None:  # 새로운 액션 샘플링
            # 1. 소스 유닛 마스크를 기반으로 소스 유닛 선택
            source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(args.num_envs, -1))
            multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
            action_components = [multi_categoricals[0].sample()]
            
            # 2. 액션 타입과 매개변수 섹션을 소스 유닛의 액션 마스크를 기반으로 선택
            source_unit_action_mask = torch.Tensor(
                np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(args.num_envs, -1))
            
            # 유닛 액션 마스크 제거 (마스킹 없음 버전이므로)
            source_unit_action_mask[:] = 1
            split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
            multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)]
            invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
            action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
            action = torch.stack(action_components)
        else:  # 기존 액션 사용
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
        
        # 로그 확률과 엔트로피 계산
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0), invalid_action_masks

    def get_value(self, x):
        """상태 가치 함수 계산"""
        return self.critic(self.forward(x))

# ==================== 에이전트 및 옵티마이저 초기화 ====================

agent = Agent().to(device)  # 에이전트를 디바이스로 이동
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  # Adam 옵티마이저

# 학습률 스케줄링 설정
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ==================== 훈련 데이터 저장소 초기화 ====================

# 에포크 데이터를 저장할 텐서들 초기화
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + (envs.action_space.nvec.sum(),)).to(device)

# ==================== 훈련 루프 초기화 ====================

global_step = 0  # 전역 스텝 카운터
start_time = time.time()  # 시작 시간 기록

# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = envs.reset()  # 초기 관찰
next_done = torch.zeros(args.num_envs).to(device)  # 초기 완료 상태
num_updates = args.total_timesteps // args.batch_size  # 총 업데이트 수

# ==================== 크래시 및 재시작 로직 ====================

starting_update = 1  # 시작 업데이트 번호
if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get('charts/update') + 1  # 마지막 업데이트 번호 + 1
    global_step = starting_update * args.batch_size  # 전역 스텝 업데이트
    api = wandb.Api()
    run = api.run(f"{run.entity}/{run.project}/{run.id}")
    model = run.file('agent.pt')  # 저장된 모델 파일 가져오기
    model.download(f"models/{experiment_name}/")  # 모델 다운로드
    agent.load_state_dict(torch.load(f"models/{experiment_name}/agent.pt", map_location=device))  # 모델 로드
    agent.eval()  # 평가 모드로 설정
    print(f"resumed at update {starting_update}")

# ==================== 메인 훈련 루프 ====================

for update in range(starting_update, num_updates+1):
    # 학습률 스케줄링 (선택적)
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates  # 남은 진행률 계산
        lrnow = lr(frac)  # 현재 학습률 계산
        optimizer.param_groups[0]['lr'] = lrnow  # 옵티마이저 학습률 업데이트

    # ==================== 데이터 수집 루프 ====================
    
    for step in range(0, args.num_steps):
        envs.render()  # 환경 렌더링
        global_step += 1 * args.num_envs  # 전역 스텝 업데이트
        obs[step] = next_obs  # 현재 관찰 저장
        dones[step] = next_done  # 현재 완료 상태 저장

        # ==================== 액션 선택 ====================
        
        with torch.no_grad():
            values[step] = agent.get_value(obs[step]).flatten()  # 상태 가치 계산
            action, logproba, _, invalid_action_masks[step] = agent.get_action(obs[step], envs=envs)

        actions[step] = action.T  # 선택된 액션 저장
        logprobs[step] = logproba  # 로그 확률 저장

        # ==================== 액션 실행 및 환경 스텝 ====================
        
        next_obs, rs, ds, infos = envs.step(action.T)  # 환경 스텝 실행
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)  # 보상과 완료 상태 저장

        # ==================== 로깅 ====================
        
        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                for key in info['microrts_stats']:
                    writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
                break

    # ==================== PPO 업데이트 ====================
    
    # 배치 한계에 도달했을 때 부트스트랩 보상 계산
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)  # 마지막 상태의 가치
        
        if args.gae:  # GAE (Generalized Advantage Estimation) 사용
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:  # 일반적인 리턴 계산
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t+1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

    # ==================== 배치 데이터 평면화 ====================
    
    b_obs = obs.reshape((-1,)+envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,)+envs.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_invalid_action_masks = invalid_action_masks.reshape((-1, invalid_action_masks.shape[-1]))

    # ==================== 정책 및 가치 네트워크 최적화 ====================
    
    target_agent = Agent().to(device)  # 타겟 에이전트 생성
    inds = np.arange(args.batch_size,)
    for i_epoch_pi in range(args.update_epochs):  # PPO 업데이트 에포크
        np.random.shuffle(inds)  # 미니배치 순서 섞기
        target_agent.load_state_dict(agent.state_dict())  # 타겟 에이전트 업데이트
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            
            # 어드밴티지 정규화 (선택적)
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # 새로운 로그 확률과 엔트로피 계산
            _, newlogproba, entropy, _ = agent.get_action(
                b_obs[minibatch_ind],
                b_actions.long()[minibatch_ind].T,
                b_invalid_action_masks[minibatch_ind],
                envs)
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()  # 중요도 샘플링 비율

            # ==================== 통계 계산 ====================
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()  # KL 발산 근사

            # ==================== 정책 손실 계산 ====================
            pg_loss1 = -mb_advantages * ratio  # 기본 정책 그래디언트 손실
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)  # 클리핑된 손실
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # PPO 클리핑 손실
            entropy_loss = entropy.mean()  # 엔트로피 손실

            # ==================== 가치 손실 계산 ====================
            new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)  # 새로운 가치 예측
            if args.clip_vloss:  # 가치 함수 클리핑 사용
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:  # 일반적인 MSE 손실
                v_loss = 0.5 *((new_values - b_returns[minibatch_ind]) ** 2)

            # ==================== 총 손실 계산 및 역전파 ====================
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef  # 총 손실

            optimizer.zero_grad()  # 그래디언트 초기화
            loss.backward()  # 역전파
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # 그래디언트 클리핑
            optimizer.step()  # 옵티마이저 스텝

        # ==================== KL 발산 기반 조기 정지 ====================
        
        if args.kle_stop:  # KL 발산 기반 조기 정지
            if approx_kl > args.target_kl:
                break
        if args.kle_rollback:  # KL 발산 기반 롤백
            if (b_logprobs[minibatch_ind] - agent.get_action(
                    b_obs[minibatch_ind],
                    b_actions.long()[minibatch_ind].T,
                    b_invalid_action_masks[minibatch_ind],
                    envs)[1]).mean() > args.target_kl:
                agent.load_state_dict(target_agent.state_dict())  # 이전 정책으로 롤백
                break

    # ==================== 모델 저장 및 로깅 ====================
    
    # 프로덕션 모드에서 모델 저장
    if args.prod_mode:
        if not os.path.exists(f"models/{experiment_name}"):
            os.makedirs(f"models/{experiment_name}")
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"agent.pt")
        else:
            if update % CHECKPOINT_FREQUENCY == 0:
                torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")

    # ==================== 훈련 통계 로깅 ====================
    
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))

# ==================== 정리 ====================

envs.close()  # 환경 종료
writer.close()  # TensorBoard writer 종료
