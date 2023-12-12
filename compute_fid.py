# from google.colab.patches import cv2_imshow
# from google.colab import auth
import gspread
import pandas as pd
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from skimage import io, transform
from tqdm import tqdm
import numpy as np
import math
# auth.authenticate_user()
# from google.auth import default
# creds, _ = default()
# auth = gspread.authorize(creds)

# worksheet = auth.open('hands and palms loras').worksheet('scale1').get("A2:I625")
# df = pd.DataFrame.from_records(worksheet)
# df = df.rename({0:'rank', 1:'checkpoint', 2:'side',3:'gender', 4:'hand', 5:'image', 6: 'bad', 7:'ok', 8:'good'}, axis=1)
# df
df = pd.read_csv('/Users/matthewhayes/Downloads/hands and palms loras - scale1.csv')

print(df)

import torch_fidelity
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision
import torch
import os
from tqdm import tqdm
import random
import gc
from torch_fidelity.fidelity import main

from functools import partial

def getpath(row):
  if not math.isnan(row['rank']):
    return f'/Users/matthewhayes/Library/CloudStorage/GoogleDrive-mjh388@cornell.edu/My Drive/LoRA (1)/inference/hands and palms r{int(row["rank"])} lora/{row["side"]}_of_fair-skinned_{row["gender"]}_{row["hand"]}_hand_chkpt{row["checkpoint"]}_steps30_{row["image"]}.png'
  else:
    return f'/Users/matthewhayes/Library/CloudStorage/GoogleDrive-mjh388@cornell.edu/My Drive/LoRA (1)/inference/hands and palms rNone lora/{row["side"]}_of_fair-skinned_{row["gender"]}_{row["hand"]}_hand_chkpt5000_steps30_{row["image"]}.png'


from torchmetrics.image.fid import FrechetInceptionDistance

dataset_path = '/Users/matthewhayes/Library/CloudStorage/GoogleDrive-mjh388@cornell.edu/My Drive/cs236 project/hands and palms/Hands/Hands'
real_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path) if not x.endswith('.csv')])
random.seed(1234)
_ = torch.manual_seed(1234)
random.shuffle(real_paths)
real_paths = real_paths[:128]
real_images = []
for path in tqdm(real_paths):
  real_images.append(torch.tensor(np.array(Image.open(path).convert("RGB").resize((299, 299)))).unsqueeze(0).permute(0, 3, 1, 2))
gc.collect()
real_images = torch.cat(real_images)
print(f"real images shape={real_images.shape}")
# real_images.to('cuda')

results = {"checkpoint":[], "r4":[], "r64":[]}
for checkpoint in [0, 5000, 10000, 15000, 20000, 25000, 30000]:
  for rank in [4, 64]:
    if checkpoint ==0 and rank == 64:
      continue
    if checkpoint ==0:
      samples = df.loc[(df['checkpoint'] == checkpoint)]
    else:
      samples = df.loc[(df['checkpoint'] == checkpoint) & (df['rank'] == rank)]
    paths = [getpath(row) for (index, row) in samples.iterrows()]
    print(f"checkpoint {checkpoint} paths size = {len(paths)}")
    images = torch.cat([torch.tensor(np.array(Image.open(path).convert("RGB"))).unsqueeze(0).permute(0, 3, 1, 2) for path in paths])
    print(f"images shape={images.shape}")
    # images.to('cuda')
    kid = FrechetInceptionDistance() # feature=192, 
    gc.collect()
    kid.update(real_images, real=True)
    gc.collect()
    kid.update(images, real=False)
    gc.collect()
    computed = kid.compute()
    print(f"checkpoint={checkpoint}, rank={rank}, kid={computed}")
    results[f'r{rank}'].append(computed.item())
    if rank == 4:
      results[f'checkpoint'].append(checkpoint)
    if checkpoint == 0:
      results[f'r64'].append(computed.item())
    # df.loc[index, 'clip'] = calculate_clip_score(image, [f'{row["side"]} of fair-skinned {row["gender"]} {row["hand"]} hand'])

print(pd.DataFrame(results))