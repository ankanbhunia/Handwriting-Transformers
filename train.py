import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["WANDB_API_KEY"] = ""

from pathlib import Path
import time
from data.dataset import TextDataset, TextDatasetval
from models import create_model
import torch
import cv2
import os
import numpy as np
from itertools import cycle
from scipy import linalg
from models.model import TRGAN
from params import *
from torch import nn
import wandb

def main():

    wandb.init(project="hwt-final", name = EXP_NAME)

    init_project()

    TextDatasetObj = TextDataset(num_examples = NUM_EXAMPLES)
    dataset = torch.utils.data.DataLoader(
                TextDatasetObj,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObj.collate_fn)

    TextDatasetObjval = TextDatasetval(num_examples = NUM_EXAMPLES)
    datasetval = torch.utils.data.DataLoader(
                TextDatasetObjval,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObjval.collate_fn)

    model = TRGAN()

    os.makedirs('saved_models', exist_ok = True)
    MODEL_PATH = os.path.join('saved_models', EXP_NAME)
    if os.path.isdir(MODEL_PATH) and RESUME: 
        model.load_state_dict(torch.load(MODEL_PATH+'/model.pth'))
        print (MODEL_PATH+' : Model loaded Successfully')
    else: 
        if not os.path.isdir(MODEL_PATH): os.mkdir(MODEL_PATH)


    for epoch in range(EPOCHS):    

        
        start_time = time.time()
        
        for i,data in enumerate(dataset): 

            if (i % NUM_CRITIC_GOCR_TRAIN) == 0:

                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DOCR_TRAIN) == 0:

                model._set_input(data)
                model.optimize_D_OCR()
                model.optimize_D_OCR_step()

            if (i % NUM_CRITIC_GWL_TRAIN) == 0:

                model._set_input(data)
                model.optimize_G_WL()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DWL_TRAIN) == 0:

                model._set_input(data)
                model.optimize_D_WL()
                model.optimize_D_WL_step()

        end_time = time.time()
        data_val = next(iter(datasetval))
        losses = model.get_current_losses()
        page = model._generate_page(model.sdata, model.input['swids'])
        page_val = model._generate_page(data_val['simg'].to(DEVICE), data_val['swids'])

        
        wandb.log({'loss-G': losses['G'],
                    'loss-D': losses['D'], 
                    'loss-Dfake': losses['Dfake'],
                    'loss-Dreal': losses['Dreal'],
                    'loss-OCR_fake': losses['OCR_fake'],
                    'loss-OCR_real': losses['OCR_real'],
                    'loss-w_fake': losses['w_fake'],
                    'loss-w_real': losses['w_real'],
                    'epoch' : epoch,
                    'timeperepoch': end_time-start_time,
                    
                    })

                    
        
        wandb.log({ "result":[wandb.Image(page, caption="page"),wandb.Image(page_val, caption="page_val")],
                    })

        

        print ({'EPOCH':epoch, 'TIME':end_time-start_time, 'LOSSES': losses})

        if epoch % SAVE_MODEL == 0: torch.save(model.state_dict(), MODEL_PATH+ '/model.pth')
        if epoch % SAVE_MODEL_HISTORY == 0: torch.save(model.state_dict(), MODEL_PATH+ '/model'+str(epoch)+'.pth')


if __name__ == "__main__":
    
    main()
