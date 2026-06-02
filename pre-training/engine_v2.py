# RNA 인코더로의 forwarding은 한번만, 이때 RNA는 30% 랜덤하게 마스킹, CLS 토큰 사용
import math
import sys
import random
from typing import Iterable, Optional

import torch
import torch.nn.functional ad F

import utils.mics as mics
import utils.lr_sched as lr_sched
import logging

logger = logging.getLogger(__name__)

def train_one_epoch(model: torch.mm.)