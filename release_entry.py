"""Explicit Lightning V10 profile; ignore inherited model/config settings."""
import os
import sys
for name in list(os.environ):
    if name.startswith('KINDI_') and not name.startswith('KINDI_LIGHTNING_'):os.environ.pop(name)
os.environ.setdefault('KINDI_LIGHTNING_KERNELS','optimized')
os.environ.setdefault('KINDI_LIGHTNING_DENSE_PREFILL','tile24')
os.environ.setdefault('KINDI_LIGHTNING_EXPERT_PREFILL','tile24')
os.environ.setdefault('KINDI_LIGHTNING_ATTENTION','split')
os.environ.setdefault('KINDI_LIGHTNING_PREFILL_CHUNK','256')
os.environ.setdefault('KINDI_LIGHTNING_PREFILL_Q2','or_aos_lb4')
os.environ.setdefault('KINDI_LIGHTNING_ATTENTION_KV_LOAD','load4')
os.environ.setdefault('KINDI_LIGHTNING_Q2_INPUT','vec4')
os.environ.setdefault('KINDI_LIGHTNING_Q2_DECODE','or')
os.environ.setdefault('KINDI_LIGHTNING_SSM_STRATEGY','auto')
os.environ.setdefault('KINDI_LIGHTNING_HEAD_LAYOUT','split_lane4')
os.environ.setdefault('KINDI_LIGHTNING_KV_READ','direct')
os.environ.setdefault('KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY','gqa16_vec2_32_bank4')
os.environ.setdefault('KINDI_LIGHTNING_ATTENTION_WORKSPACE_MB','512')
os.makedirs('/workspace/state/kernels/driver',exist_ok=True)
os.execv(sys.executable,[sys.executable,'-m','kindi','--config','/release/config/kindi.conf']+(sys.argv[1:] or ['serve']))
