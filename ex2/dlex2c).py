
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
path = '/kaggle/input/plant-pathology-2020-fgvc7/'
img_path = path + 'images'
# LOAD THE DATASET
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample = pd.read_csv(path + 'sample_submission.csv')
# GET THE IMAGE FILE NAME
train_df['img_path'] = train_df['image_id'] + '.jpg'
test_df['img_path'] = test_df['image_id'] + '.jpg'
train_df.head()

train_label = train_df.melt(id_vars=['image_id', 'img_path'])
train_label = train_label[train_label['value'] == 1]
train_label['id'] = [int(i[1]) for i in train_label['image_id'].str.split('_')]
train_label = train_label.sort_values('id').reset_index()
train_df['label'] = train_label['variable']
train_df = train_df[train_df.columns[[0, 5, 1, 2, 3, 4, 6]]]
print(train_label.shape)
train_df.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_encoded = le.fit_transform(train_df['label'])
train_df['label_encoded'] = label_encoded
train_df.head()

from torchvision import transforms
import torch
import matplotlib.pyplot as plt
class Plants():
    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, -1]
        
        if self.transform:
            image = self.transform(image)
    
        return (image, label)
plant_train = Plants(
    data_frame=train_df,
    root_dir=path + 'images',
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)
examples = enumerate(plant_train)
examples