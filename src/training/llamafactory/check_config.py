"""Validate recipe keys against the pinned argument parser without allocating a GPU."""
import argparse
import pathlib
from train import config
from llamafactory.hparams.parser import _parse_train_args
args=argparse.Namespace(model='/model',max_length=1024,max_steps=2)
values=config(args,pathlib.Path('/out'))
# Dataclass construction otherwise validates BF16 against the CPU test host.
values.update(use_cpu=True,bf16=False)
_parse_train_args(values)
print('configuration_keys_validated')
