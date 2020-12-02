#%%
from models import Resnet18, Resnet50, EfficientNetB1
from dataloaders import load_image
import torch
import pandas as pd 
import os
import argparse
# %%
def inference(args):
    classifier_list = [Resnet18, Resnet50, EfficientNetB1]
    classifier_names = [elem.__name__.lower() for elem in classifier_list]
    classifier_model_name = args.m
    classifier = classifier_list[classifier_names.index(classifier_model_name)]
    checkpoint_dir = os.path.join('logs', args.m)
    checkpoints = [elem for elem in os.listdir(checkpoint_dir) if elem.split('.')[-1] =='ckpt']
    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
    model = classifier.load_from_checkpoint(checkpoint_path)
    model.freeze()
    df = pd.read_csv(os.path.join(args.dd, 'sample_submission.csv'))
    TEST_DIR = os.path.join(args.dd, 'test_images')
    for num in range(len(df)):
        x = load_image(os.path.join(TEST_DIR, df.loc[num, 'image_id']), to_tensor=True)
        df.loc[num, 'label'] = torch.argmax(model(x.unsqueeze(0))).cpu().numpy()
    df.to_csv('submission.csv')
    if args.tr:
        df = pd.read_csv(os.path.join(args.dd, 'train.csv'))
        TRAIN_DIR = os.path.join(args.dd, 'train_images')
        for num in range(len(df)):
            x = load_image(os.path.join(TRAIN_DIR, df.loc[num, 'image_id']), to_tensor=True)
            df.loc[num, 'label_pred'] = torch.argmax(model(x.unsqueeze(0))).cpu().numpy()
        df.to_csv('train_submission.csv')
#%%
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dd', type=str, default='..', help='Data directory')
    parser.add_argument('--m', type=str, default='efficientnetb1', help='model')
    parser.add_argument('--cp', type=str, default='', help='checkpoint path')
    parser.add_argument('--tr', type=bool, default=True, help='inference on train')
    args = parser.parse_args()
    inference(args)
