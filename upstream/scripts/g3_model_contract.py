"""G3 model contracts: freeze the structures of the three model variants.

Writes (read-only w.r.t. model code):
    results/package_baseline/g3_model_contract/package_model_contract.json
    results/package_baseline/g3_model_contract/paper_model_contract.json
    results/package_baseline/g3_model_contract/residual_model_status.json

Regenerate only when the model code or model.yaml intentionally changes.

Run (CPU is enough, soydngp312 env):
    /home/yhshy/miniconda3/envs/soydngp312/bin/python scripts/g3_model_contract.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch
import torch.nn.functional as F

from soydngpnext.remodel import remodel

B = 2
X = torch.randn(B, 3, 206, 206)
OUT = os.path.join(ROOT, 'results', 'package_baseline', 'g3_model_contract')
os.makedirs(OUT, exist_ok=True)


def child_shapes(module, x):
    """Output shape of each top-level child module, via forward hooks."""
    shapes = {}
    handles = []
    for name, child in module.named_children():
        def h(m, inp, out, name=name):
            shapes[name] = list(out.shape)
        handles.append(child.register_forward_hook(h))
    with torch.no_grad():
        module(x)
    for h in handles:
        h.remove()
    return shapes


def flatten_features(module, x):
    flat = next(m for m in module.modules() if isinstance(m, torch.nn.Flatten))
    out = {}

    def h(m, inp, o):
        out['shape'] = list(o.shape)

    handle = flat.register_forward_hook(h)
    with torch.no_grad():
        module(x)
    handle.remove()
    return out['shape'][1]


def backward_check(module, num_classes):
    module.train()
    out = module(X)
    if num_classes == 1:
        loss = F.mse_loss(out, torch.rand_like(out))
    else:
        loss = F.cross_entropy(out, torch.randint(0, num_classes, (B,)))
    loss.backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    return {
        'loss_finite': bool(torch.isfinite(loss).all()),
        'grads_finite': all(bool(torch.isfinite(g).all()) for g in grads),
        'params_with_grad': len(grads),
    }


def param_counts(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def blocks(module):
    return [[name, type(child).__name__] for name, child in module.named_children()]


# ---------------------------------------------------------------- package
print('== G3-A package yaml model ==')
YAML = os.path.join(ROOT, 'soydngpnext', 'data', 'model.yaml')
net, _ = remodel(YAML, 1, show_structure=False)
net.eval()
with torch.no_grad():
    out_reg = net(X)
net_c, _ = remodel(YAML, 9, show_structure=False)
net_c.eval()
with torch.no_grad():
    out_cls = net_c(X)

total, trainable = param_counts(net)
pkg = {
    'model': 'package yaml model',
    'source': 'soydngpnext/data/model.yaml @ upstream-legacy (ebda01d)',
    'built_with': "remodel('soydngpnext/data/model.yaml', num_classes)",
    'input_shape': [B, 3, 206, 206],
    'regression': {
        'num_classes': 1,
        'output_shape': list(out_reg.shape),
        'output_finite': bool(torch.isfinite(out_reg).all()),
        'backward': backward_check(net, 1),
    },
    'classification': {
        'num_classes': 9,
        'output_shape': list(out_cls.shape),
        'output_finite': bool(torch.isfinite(out_cls).all()),
        'backward': backward_check(net_c, 9),
    },
    'intermediate_shapes': child_shapes(net, X),
    'flatten_features': flatten_features(net, X),
    'total_parameters': total,
    'trainable_parameters': trainable,
    'module_names': blocks(net),
    'state_dict_keys': sorted(net.state_dict().keys()),
}
with open(os.path.join(OUT, 'package_model_contract.json'), 'w') as f:
    json.dump(pkg, f, indent=2)
print('regression output:', pkg['regression']['output_shape'],
      '| classification output:', pkg['classification']['output_shape'])
print('flatten_features:', pkg['flatten_features'],
      '| total params:', total, '| trainable:', trainable)
print('PASS G3-A package model contract written')

# ----------------------------------------------------------------- paper
print('== G3-B paper CA model ==')
from paper_model.soydngp_ca import AlexNet, CA_Block

m = AlexNet()
m.net.eval()
with torch.no_grad():
    out_paper = m.net(X)

conv_idx = [i for i, mod in enumerate(m.net) if isinstance(mod, torch.nn.Conv2d)]
ca_idx = [i for i, mod in enumerate(m.net) if isinstance(mod, CA_Block)]
total_p, trainable_p = param_counts(m)
paper = {
    'model': 'paper model (PAPER_MODEL_V1)',
    'source': {
        'repo': 'IndigoFloyd/SoybeanWebsite',
        'file': 'AlexNet_206.py',
        'commit': 'f724d9b1974bb78b7832b191af99f589a2c5e549',
        'fetched': '2026-09-11',
        'port': 'src/paper_model/soydngp_ca.py',
        'policy': 'verbatim copy, no refactoring',
    },
    'architecture': {
        'conv_layers': len(conv_idx),
        'conv_layer_indices': conv_idx,
        'ca_blocks': [
            {'index': ca_idx[0], 'channels': 32, 'spatial': [206, 206],
             'position': 'after first conv stage'},
            {'index': ca_idx[1], 'channels': 1024, 'spatial': [7, 7],
             'position': 'before flatten'},
        ],
        'flatten_features': 50176,
        'head': 'Linear(50176, 1, bias=True)',
    },
    'input_shape': [B, 3, 206, 206],
    'regression': {
        'output_shape': list(out_paper.shape),
        'output_finite': bool(torch.isfinite(out_paper).all()),
        'backward': backward_check(m.net, 1),
    },
    'intermediate_shapes': child_shapes(m.net, X),
    'total_parameters': total_p,
    'trainable_parameters': trainable_p,
    'module_names': blocks(m.net),
    'state_dict_keys': sorted(m.state_dict().keys()),
    'quirks': [
        "AlexNet.modules() is overridden to return (self.net, 'AlexNet_deep'); "
        'anything calling .modules()/.named_modules() on the top module breaks; '
        'children()-based paths (to/eval/train/apply/state_dict) are unaffected',
        'no forward() on AlexNet; inference goes through model.net',
        'final Linear has bias (package model uses bias=False)',
    ],
}
with open(os.path.join(OUT, 'paper_model_contract.json'), 'w') as f:
    json.dump(paper, f, indent=2)
print('output:', paper['regression']['output_shape'],
      '| conv layers:', len(conv_idx), '| CA at', ca_idx,
      '| total params:', total_p)
print('PASS G3-B paper model contract written')

# --------------------------------------------------------------- residual
print('== G3-C 2025 residual variant ==')
from soydngpnext.SoyDNGP_res import SoyDNGP

res = {
    'model': 'soydngpnext/SoyDNGP_res.py :: SoyDNGP',
    'MODEL_VARIANT': 'POST_PUBLICATION_RESIDUAL',
    'REPRODUCTION_ROLE': 'SECONDARY',
    'import': 'ok',
    'instantiate': 'ok: SoyDNGP(dropout=0.3)',
}
rm = SoyDNGP(0.3)
rm.eval()
try:
    with torch.no_grad():
        out_res = rm(torch.randn(1, 3, 206, 206))
    res['forward_on_expected_input'] = {
        'input_shape': [1, 3, 206, 206],
        'result': 'ok',
        'output_shape': list(out_res.shape),
    }
except Exception as e:
    res['forward_on_expected_input'] = {
        'input_shape': [1, 3, 206, 206],
        'result': '%s: %s' % (type(e).__name__, e),
        'finding': ('ca1 = CA_Block(32, 322, 322): h/w=322 exceeds the 206x206 '
                    'spatial size, expand_as fails; not executable on the '
                    'expected input, excluded from experiments'),
    }
res['excluded_from_experiments'] = True
with open(os.path.join(OUT, 'residual_model_status.json'), 'w') as f:
    json.dump(res, f, indent=2)
print('forward on (1,3,206,206):', res['forward_on_expected_input']['result'])
print('PASS G3-C residual status written')
