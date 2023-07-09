import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk
import numpy as np
import torch

          
class trainerData3d_preload(Dataset):
    def __init__(self, img_path, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_path = img_path
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
        self.all_image_data = []
        for index in range(len(self.img_path)):
    
            get_img = sitk.ReadImage('../../../../' + self.img_path[index]+'/Img_final_0.nii.gz')
            return_img = sitk.GetArrayFromImage(get_img).astype(np.float32)

            num_index = len(return_img) // 2
            return_img = return_img[num_index-10:num_index+10]
            return_img = return_img[np.newaxis,:,:,:]
            self.all_image_data.append(return_img)
         
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()
        
        return_img = self.all_image_data[index]
        return_img = torch.from_numpy(return_img).float().cuda()
        return return_data, return_yt, return_img
    def __len__(self):
        return len(self.img_path)        



class trainerData_cli(Dataset):
    def __init__(self, img_path, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_path = img_path
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()
        #return_outcome = torch.from_numpy(self.outcome[index]).float().cuda()
        #return_treatment = torch.from_numpy(self.return_treatment[index]).float().cuda()
        
        
        return return_data, return_yt
    def __len__(self):
        return len(self.img_path)    


class trainerData(Dataset):
    def __init__(self, img_path, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_path = img_path
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()
        #return_outcome = torch.from_numpy(self.outcome[index]).float().cuda()
        #return_treatment = torch.from_numpy(self.return_treatment[index]).float().cuda()
        try:
            get_img = sitk.ReadImage('../../../../' + self.img_path[index]+'/Img_final_0.nii.gz')
            return_img = sitk.GetArrayFromImage(get_img).astype(np.float32)
            if return_img.shape[0] < 14:
                print(self.img_path[index])
                return_img = np.zeros((25,224,224))
        except:
            return_img = np.zeros((25,224,224))
        num_index = len(return_img) // 2
        return_img = torch.from_numpy(return_img[num_index - 2: num_index + 1]).float().cuda()
        return return_data, return_yt, return_img
    def __len__(self):
        return len(self.img_path)


class trainerData_single(Dataset):
    def __init__(self, img_path, data, outcome, treatment, is_train = True):
        self.is_train = is_train
        self.img_path = img_path
        self.data = data
        self.outcome = outcome
        self.treatment = treatment
    def __getitem__(self, index):
        return_data = torch.from_numpy(self.data[index]).float().cuda()
        return_yt = torch.from_numpy(np.concatenate([self.outcome[index], self.treatment[index]], 0)).float().cuda()
        #return_outcome = torch.from_numpy(self.outcome[index]).float().cuda()
        #return_treatment = torch.from_numpy(self.return_treatment[index]).float().cuda()
        
        return return_data, return_yt
    def __len__(self):
        return len(self.img_path)

def convert_file(x):
    x = x.values
    x = x.astype(float)
    return x


def load_and_format_covariates(file_path):

    data = pd.read_excel(file_path)

    data = data.values[1:, ]

    #binfeats = list(range(6,37))
    #contfeats = [i for i in range(37) if i not in binfeats]

    mu_0, mu_1, path, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5], data[:, 6:]
    #perm = binfeats
    #x = x[:, perm].astype(float)

    # for num in range(len(x)):
        # a1 = x[num].astype(float)
        # print(num)

    return x.astype(float), path


def load_all_other_crap(file_path):
    data = pd.read_excel(file_path)
    data = data.values[1:, ]
    t, y, y_cf = data[:, 0], data[:, 1][:, None], data[:, 2][:, None]
    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 6:]
    return t.reshape(-1, 1).astype(float), y.astype(float), y_cf.astype(float), mu_0.astype(float), mu_1.astype(float)

def main():
    pass


if __name__ == '__main__':
    main()
