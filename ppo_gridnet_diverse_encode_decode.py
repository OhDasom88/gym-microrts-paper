"""
PPO GridNet Diverse Encode-Decode 에이전트 훈련 스크립트

이 스크립트는 GridNet 아키텍처를 사용하여 다양한 AI와 대전하는 PPO 에이전트를 훈련합니다.
인코더-디코더 구조를 사용하여 그리드 기반 액션 선택을 수행합니다.

주요 특징:
- GridNet 아키텍처 (각 그리드 셀에 대해 액션 예측)
- 다양한 AI와의 대전 (CoacAI, RandomBiasedAI, LightRushAI, WorkerRushAI)
- 인코더-디코더 구조 사용
- 16x16 그리드 기반 액션 공간

참고 논문: http://proceedings.mlr.press/v97/han19a/han19a.pdf
"""

# ==================== 라이브러리 임포트 ====================
# PyTorch 관련
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# 기본 라이브러리
import argparse
from distutils.util import strtobool
import numpy as np
import time
import random
import os

# Gym 및 MicroRTS 관련
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space

# 환경 래퍼
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder

if __name__ == "__main__":
    # ==================== 명령행 인자 파싱 ====================
    parser = argparse.ArgumentParser(description='PPO agent')
    
    # 공통 인자들
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicrortsDefeatCoacAIShaped-v3",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # PPO 알고리즘 특화 인자들
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-bot-envs', type=int, default=24,
                        help='the number of bot game environment; 16 bot envs measn 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=0,
                        help='the number of self play envs; 16 self play envs means 8 games')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    # 인자 파싱 및 기본값 설정
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# 환경 및 배치 크기 계산
args.num_envs = args.num_selfplay_envs + args.num_bot_envs
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
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

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


class MicroRTSStatsRecorder(VecEnvWrapper):
    """
    MicroRTS 특화 통계를 기록하는 VecEnv 래퍼
    
    각 에피소드의 원시 보상(raw rewards)을 추적하고,
    에피소드 종료 시 MicroRTS 통계를 제공합니다.
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
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]  # 원시 보상 추가
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:  # 에피소드 종료 시
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)  # 원시 보상 합계
                raw_names = [str(rf) for rf in self.rfs]  # 보상 함수 이름들
                info['microrts_stats'] = dict(zip(raw_names, raw_rewards))  # MicroRTS 통계 추가
                self.raw_rewards[i] = []  # 원시 보상 기록 초기화
                newinfos[i] = info
        return obs, rews, dones, newinfos


# ==================== 환경 설정 ====================

# 실험 이름 생성 (고유성 보장)
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

# TensorBoard 로거 설정
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

# Weights & Biases 로깅 설정 (프로덕션 모드)
if args.prod_mode:
    import wandb
    run = wandb.init(
        project=args.wandb_project_name, entity=args.wandb_entity,
        # sync_tensorboard=True,
        config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 50

# ==================== 시드 설정 ====================
# 재현 가능한 결과를 위한 시드 설정
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
# ==================== MicroRTS 환경 생성 ====================
# GridNet 모드로 MicroRTS 환경 생성 (다양한 AI와 대전)
envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,  # 셀프플레이 환경 수
    num_bot_envs=args.num_bot_envs,  # 봇 환경 수
    max_steps=2000,  # 최대 스텝 수
    render_theme=2,  # 렌더링 테마
    # 다양한 AI 상대 (CoacAI, RandomBiasedAI, LightRushAI, WorkerRushAI)
    ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)] + \
         [microrts_ai.randomBiasedAI for _ in range(min(args.num_bot_envs, 2))] + \
         [microrts_ai.lightRushAI for _ in range(min(args.num_bot_envs, 2))] + \
         [microrts_ai.workerRushAI for _ in range(min(args.num_bot_envs, 2))],
    map_path="maps/16x16/basesWorkers16x16.xml",  # 16x16 맵
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])  # 보상 가중치
)

# 환경 래퍼 적용
envs = MicroRTSStatsRecorder(envs, args.gamma)  # MicroRTS 통계 기록
envs = VecMonitor(envs)  # 에피소드 모니터링

# 비디오 캡처 설정 (옵션)
if args.capture_video:
    envs = VecVideoRecorder(envs, f'videos/{experiment_name}',
                            record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)

# 액션 공간 검증 (MultiDiscrete만 지원)
assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"


# ==================== 에이전트 아키텍처 ====================

class CategoricalMasked(Categorical):
    """
    액션 마스킹을 지원하는 Categorical 분포
    
    유효하지 않은 액션을 마스킹하여 선택할 수 없도록 합니다.
    GridNet에서 각 그리드 셀에 대해 유효한 액션만 선택할 수 있게 합니다.
    """
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], sw=None):
        self.masks = masks
        if len(self.masks) == 0:
            # 마스킹이 없는 경우 일반 Categorical 분포 사용
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            # 마스킹이 있는 경우 유효하지 않은 액션의 로짓을 매우 작은 값으로 설정
            self.masks = masks.bool()
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8, device=device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        """마스킹을 고려한 엔트로피 계산"""
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)


class Scale(nn.Module):
    """입력에 스케일을 적용하는 모듈"""
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Transpose(nn.Module):
    """텐서 차원을 재배열하는 모듈"""
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """레이어 가중치를 직교 초기화하고 편향을 상수로 설정"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Encoder(nn.Module):
    """
    인코더: 관찰을 특징 벡터로 변환
    
    CNN 기반 인코더로 관찰을 압축된 특징 표현으로 변환합니다.
    """
    def __init__(self, input_channels):
        super().__init__()
        self._encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),  # BHWC -> BCHW
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
    """
    디코더: 특징 벡터를 그리드 형태로 변환
    
    인코더의 출력을 받아 원본 그리드 크기로 복원하고
    각 그리드 셀에 대한 액션 로짓을 생성합니다.
    """
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
            Transpose((0, 2, 3, 1)),  # BCHW -> BHWC
        )

    def forward(self, x):
        return self.deconv(x)


class Agent(nn.Module):
    """
    GridNet PPO 에이전트 (인코더-디코더 구조)
    
    인코더-디코더 아키텍처를 사용하여 각 그리드 셀에 대해 액션을 예측합니다.
    인코더는 관찰을 특징으로 변환하고, 디코더는 특징을 그리드 형태의 액션으로 변환합니다.
    """
    def __init__(self, mapsize=16 * 16):
        super(Agent, self).__init__()
        self.mapsize = mapsize  # 16x16 = 256 그리드 셀
        h, w, c = envs.observation_space.shape

        # 인코더: 관찰을 특징으로 변환
        self.encoder = Encoder(c)

        # 액터: 특징을 그리드 형태의 액션 로짓으로 변환
        self.actor = Decoder(78)  # 78개 액션 파라미터

        # 크리틱: 특징을 상태 가치로 변환
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(256, 128), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        )

    def forward(self, x):
        """관찰을 특징으로 변환"""
        return self.encoder(x)  # "bhwc" -> "bchw"

    def get_action(self, x, action=None, invalid_action_masks=None, envs=None):
        """액션 선택 및 로그 확률 계산"""
        # 인코더를 통해 특징 추출 후 디코더로 액션 로짓 생성
        logits = self.actor(self.forward(x))
        grid_logits = logits.reshape(-1, envs.action_space.nvec[1:].sum())
        split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

        if action is None:
            # 액션 샘플링
            invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                     dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            # 기존 액션에 대한 로그 확률 계산
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                     dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
        
        # 로그 확률과 엔트로피 계산
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        
        # 그리드 형태로 재구성
        num_predicted_parameters = len(envs.action_space.nvec) - 1
        logprob = logprob.T.view(-1, 256, num_predicted_parameters)
        entropy = entropy.T.view(-1, 256, num_predicted_parameters)
        action = action.T.view(-1, 256, num_predicted_parameters)
        invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

    def get_value(self, x):
        """상태 가치 예측"""
        return self.critic(self.forward(x))


# ==================== 에이전트 및 옵티마이저 초기화 ====================
agent = Agent().to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# 학습률 어닐링 설정
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ==================== 에포크 데이터 저장소 초기화 ====================
mapsize = 16 * 16  # 16x16 그리드 크기
action_space_shape = (mapsize, envs.action_space.shape[0] - 1)  # 액션 공간 형태
invalid_action_shape = (mapsize, envs.action_space.nvec[1:].sum() + 1)  # 무효 액션 마스크 형태

# 각 스텝별 데이터 저장소
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)

# ==================== 훈련 루프 시작 ====================
global_step = 0
start_time = time.time()

# 다음 관찰과 완료 상태 초기화
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

# ==================== 크래시 및 재시작 로직 ====================
starting_update = 1
from jpype.types import JArray, JInt

# Weights & Biases에서 재시작된 경우 모델 로드
if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get('charts/update') + 1
    global_step = starting_update * args.batch_size
    api = wandb.Api()
    run = api.run(f"{run.entity}/{run.project}/{run.id}")
    model = run.file('agent.pt')
    model.download(f"models/{experiment_name}/")
    agent.load_state_dict(torch.load(f"models/{experiment_name}/agent.pt", map_location=device))
    agent.eval()
    print(f"resumed at update {starting_update}")

# ==================== 모델 정보 출력 ====================
print("Model's state_dict:")
for param_tensor in agent.state_dict():
    print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
total_params = sum([param.nelement() for param in agent.parameters()])
print("Model's total parameters:", total_params)

# ==================== 메인 훈련 루프 ====================
for update in range(starting_update, num_updates + 1):
    # 학습률 어닐링 (선택사항)
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    # ==================== 데이터 수집 단계 ====================
    for step in range(0, args.num_steps):
        envs.render()  # 환경 렌더링
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
        
        # 액션 선택 및 가치 예측
        with torch.no_grad():
            values[step] = agent.get_value(obs[step]).flatten()
            action, logproba, _, invalid_action_masks[step] = agent.get_action(obs[step], envs=envs)

        actions[step] = action
        logprobs[step] = logproba

        # 실제 액션 생성 (소스 유닛 추가)
        # the real action adds the source units
        real_action = torch.cat([
            torch.stack(
                [torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)
                 ]).unsqueeze(2), action], 2)

        # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
        # so as to predict an action for each cell in the map; this obviously include a
        # lot of invalid actions at cells for which no source units exist, so the rest of
        # the code removes these invalid actions to speed things up
        real_action = real_action.cpu().numpy()
        valid_actions = real_action[invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
        valid_actions_counts = invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()
        
        # Java 배열로 변환 (MicroRTS 환경 호환성)
        java_valid_actions = []
        valid_action_idx = 0
        for env_idx, valid_action_count in enumerate(valid_actions_counts):
            java_valid_action = []
            for c in range(valid_action_count):
                java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
                valid_action_idx += 1
            java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
        java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)

        # 환경에서 스텝 실행 (Java 배열 사용)
        try:
            next_obs, rs, ds, infos = envs.step(java_valid_actions)
            next_obs = torch.Tensor(next_obs).to(device)
        except Exception as e:
            e.printStackTrace()
            raise
        rewards[step], next_done = torch.Tensor(rs).to(device), torch.Tensor(ds).to(device)

        # 에피소드 완료 시 로깅
        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                for key in info['microrts_stats']:
                    writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
                break

    # ==================== PPO 업데이트 ====================
    # 배치 한계에 도달한 경우 부트스트랩 보상 계산
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
        if args.gae:
            # GAE (Generalized Advantage Estimation) 사용
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            # 일반적인 반환값 계산
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

    # 배치 데이터 평탄화
    b_obs = obs.reshape((-1,) + envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + action_space_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)

    # ==================== 정책 및 가치 네트워크 최적화 ====================
    inds = np.arange(args.batch_size, )
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            
            # 어드밴티지 정규화
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # 새로운 액션 로그 확률 계산
            _, newlogproba, entropy, _ = agent.get_action(
                b_obs[minibatch_ind],
                b_actions.long()[minibatch_ind],
                b_invalid_action_masks[minibatch_ind],
                envs)
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            # KL 발산 통계
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            # 정책 손실 (PPO 클리핑)
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # 가치 손실
            new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef,
                                                                  args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

            # 총 손실 계산
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    # ==================== 크래시 및 재시작 로직 ====================
    if args.prod_mode:
        if not os.path.exists(f"models/{experiment_name}"):
            os.makedirs(f"models/{experiment_name}")
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"agent.pt")
        else:
            if update % CHECKPOINT_FREQUENCY == 0:
                torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")

    # ==================== 로깅 및 통계 기록 ====================
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
envs.close()
writer.close()
