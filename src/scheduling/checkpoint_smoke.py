"""CPU Trainer stop/save/resume acceptance using the production callback."""
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
from transformers import Trainer, TrainingArguments
sys.path.insert(0,'/workspace/training')
from nightly import nightly_callback, latest_complete_checkpoint

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear=torch.nn.Linear(1,1)
    def forward(self,input_ids,labels):
        y=self.linear(input_ids.float()); return {'loss':((y-labels)**2).mean(),'logits':y}

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); flag=root/'stop'; flag.touch(); os.environ['RTL_STOP_FILE']=str(flag)
    args=TrainingArguments(output_dir=str(root/'run'),max_steps=3,per_device_train_batch_size=1,save_steps=2,report_to='none',use_cpu=True)
    data=[{'input_ids':torch.tensor([1.]),'labels':torch.tensor([2.])} for _ in range(4)]
    trainer=Trainer(model=Model(),args=args,train_dataset=data,callbacks=[nightly_callback()]); trainer.train()
    assert trainer.state.global_step==1
    checkpoint=latest_complete_checkpoint(root/'run'); assert checkpoint and checkpoint.endswith('checkpoint-1')
    flag.unlink()
    trainer=Trainer(model=Model(),args=args,train_dataset=data,callbacks=[nightly_callback()]); trainer.train(resume_from_checkpoint=checkpoint)
    assert trainer.state.global_step==3
    assert latest_complete_checkpoint(root/'run').endswith('checkpoint-3')
    print(json.dumps({'stop_step':1,'resume_final_step':3,'atomic_checkpoint':'passed'}))
