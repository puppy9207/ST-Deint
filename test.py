import torch
import cv2
import time
import os
import numpy as np
import copy

from torchvision.transforms import ToTensor
from glob import glob
from model import Net as STNet



if __name__ == "__main__":
    indir = "image"
    outdir = "output"
    os.environ["CUDA_VISIBLE_DEVICES"]= str("3")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = STNet().to(device)
    net = torch.load("model_w_atten.pt")
    # print(net)
    # model.load_state_dict(net)
    cnt = 0
    file_cnt = 0
    for x in glob(f"{indir}/*"):
        image = cv2.imread(x)
        img_copy = copy.deepcopy(image)
        h,w,c = image.shape
        # image = image.transpose([2,0,1])
        file_name = os.path.basename(x).split(".")[0]
        image = ToTensor()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = net(image)

        restoreA, restoreB = preds
        restoreA = restoreA.mul(255).clamp(0,255).cpu().numpy().squeeze()
        restoreA = restoreA.transpose([1,2,0])
        restoreB = restoreB.mul(255).clamp(0,255).cpu().numpy().squeeze()
        restoreB = restoreB.transpose([1,2,0])
        outputA = []
        outputB = []
        for i in range(h):
            if i%2==0:
                outputA.append(restoreB[i//2,:,:])
                outputB.append(img_copy[i,:,:])
            else:
                outputA.append(img_copy[i,:,:])
                outputB.append(restoreA[i//2,:,:])

        outputA = np.asarray(outputA)
        outputB = np.asarray(outputB)
        cv2.imwrite("output/img%03d.png"%(file_cnt),outputB)
        file_cnt+=1
        cv2.imwrite("output/img%03d.png"%(file_cnt),outputA)
        file_cnt+=1