"""
모델 다운로드 스크립트

이 스크립트는 wandb에서 훈련된 모델들을 다운로드하여 로컬에 저장합니다.
각 실험의 시드별로 모델 파일을 구분하여 저장합니다.

주요 기능:
- wandb 프로젝트에서 모든 실행 결과 조회
- 각 실험의 훈련된 모델 다운로드
- 시드별로 모델 파일명 구분하여 저장

사용법:
    python download_models.py --wandb-project "your-project/your-entity"
"""

from os import path
import pickle
import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from distutils.util import strtobool

# ==================== 명령행 인수 파싱 ====================

parser = argparse.ArgumentParser(description='CleanRL Plots')

# ==================== 공통 인수 ====================

parser.add_argument('--wandb-project', type=str, default="vwxyzjn/gym-microrts-paper",
                   help='wandb 프로젝트 이름 (예: cleanrl/cleanrl)')

args = parser.parse_args()

# ==================== wandb API 초기화 ====================

api = wandb.Api()  # wandb API 인스턴스 생성
runs = api.runs(args.wandb_project)  # 지정된 프로젝트의 모든 실행 결과 조회

# ==================== 모델 다운로드 ====================

for idx, run in enumerate(runs):
    exp_name = run.config['exp_name']  # 실험 이름 가져오기
    
    # 실험별 디렉토리 생성
    if not os.path.exists(f"trained_models/{exp_name}"):
        os.makedirs(f"trained_models/{exp_name}")
    
    # 시드별 모델 파일이 존재하지 않는 경우에만 다운로드
    if not os.path.exists(f"trained_models/{exp_name}/agent-{run.config['seed']}.pt"):
        trained_model = run.file('agent.pt')  # 훈련된 모델 파일 가져오기
        trained_model.download(f"trained_models/{exp_name}")  # 모델 다운로드
        
        # 시드별로 구분된 파일명으로 변경
        os.rename(f"trained_models/{exp_name}/agent.pt",
                f"trained_models/{exp_name}/agent-{run.config['seed']}.pt")
