#%%
from models import Resnet18
from dataloaders import load_image
import torch
import pandas as pd 
import os
import argparse
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--dd', type=str, default='..', help='Data directory')
parser.add_argument('--m', type=str, default='resnet18', help='model')
args = parser.parse_args()
# %%
df = pd.read_csv(os.path.join(args.dd, 'sample_submission.csv'))
# %%
checkpoint_dir = os.path.join('logs', args.m)
checkpoints = [elem for elem in os.listdir(checkpoint_dir) if elem.split('.')[-1] =='ckpt']
checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
#%%
model = Resnet18.load_from_checkpoint(checkpoint_path)
model.freeze()

# %%
TEST_DIR = os.path.join(args.dd, 'test_images')
# %%
for num in range(len(df)):
    x = load_image(os.path.join(TEST_DIR, df.loc[num, 'image_id']))
    df.loc[num, 'label'] = torch.argmax(model(x.unsqueeze(0))).cpu().numpy()
# %%
