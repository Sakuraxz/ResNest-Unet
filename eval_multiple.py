import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from data_loading import multi_classes
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
from pytorch_lightning.metrics import ConfusionMatrix
import os 
import cv2


def get_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])
def binary_dice(inputs, targets, smooth=1):
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
    return 1-dice

# python eval_multiple.py --dataset ./data  --csvfile ./src/test_train_data.csv --result_name ver_1 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,type=str, help='the path of dataset')
    parser.add_argument('--csvfile', required=True,type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--result_name', type=str, required=True, help='result like ver_1')
    parser.add_argument('--model',default='epoch_last.pth', type=str, help='the path of model')
    parser.add_argument('--debug',default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    
    os.makedirs(f'result/{args.result_name}/img',exist_ok=False)
    df = pd.read_csv(args.csvfile)
    df = df[df.category=='test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_files = list(df.image_id)
    test_dataset = multi_classes(args.dataset,test_files, get_transform())
    model = torch.load(f'result/{args.result_name}/save_models/{args.model}')
    model = model.cuda()
    sfx = nn.Softmax(dim=1)
    cfs = ConfusionMatrix(4)
    loss_score = []
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    time_cost = []
    since = time.time()
    
    for image_id in test_files:
        img = cv2.imread(f'{args.dataset}/images/{image_id}')
        img = cv2.resize(img, (512,512))
        cv2.imwrite(f'result/{args.result_name}/img/{image_id}',img)

    with torch.no_grad():
        for img, mask, img_id in test_dataset:
            
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()           
            mask = Variable(torch.unsqueeze(mask, dim=0).float(), requires_grad=False).cuda()
            
            torch.cuda.synchronize()
            start = time.time()
            pred = model(img)
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end-start)
            
            ts_sfx = sfx(pred)
            pred = sfx(pred)
            ipts = pred.cpu()
            gt =  mask.cpu()
            img_class = torch.max(ts_sfx,1).indices.cpu()
            c = pred.shape[1]
            pred = torch.max(pred,1).indices.cpu()
            mask = torch.max(mask,1).indices.cpu()
            mask_draw = mask.clone().detach()
            
            if args.debug:
        
                img_numpy = pred.detach().numpy()[0]
                img_numpy = img_numpy.astype(np.uint8)
                img_numpy = cv2.resize(img_numpy, (512,512))
                img_numpy[img_numpy==1] = 85
                img_numpy[img_numpy==2] = 170
                img_numpy[img_numpy==3] = 255
                cv2.imwrite(f'result/{args.result_name}/img/{img_id}_pred.png',img_numpy) 
                #pred 通常是指预测值（predictions），即模型对输入数据的输出或估计值。
                
                mask_numpy = mask_draw.detach().numpy()[0]
                mask_numpy = mask_numpy.astype(np.uint8)
                mask_numpy = cv2.resize(mask_numpy, (512,512))
                mask_numpy[mask_numpy==1] = 85
                mask_numpy[mask_numpy==2] = 170
                mask_numpy[mask_numpy==3] = 255
                cv2.imwrite(f'result/{args.result_name}/img/{img_id}_gt.png',mask_numpy)
                #gt 则通常是指真实标签值（ground truth），即实际的目标数值或类别标签，用于与模型预测进行比较或评估。
               
            cfsmat = cfs(img_class,mask).numpy()
            
            sum_loss = 0
            sum_iou = 0
            sum_prec = 0
            sum_acc = 0
            sum_recall = 0
            sum_f1 = 0
            
            for i in range(c):            
                tp = cfsmat[i,i]
                fp = np.sum(cfsmat[0:c,i]) - tp
                fn = np.sum(cfsmat[i,0:c]) - tp
                
                tmp_inputs = ipts[:,i]
                tmp_gts   = gt[:,i]
                
                tmp_loss  = binary_dice(tmp_inputs,tmp_gts)            
                tmp_iou = tp / (fp + fn + tp)
                tmp_prec = tp / (fp + tp + 1) 
                tmp_acc = tp
                tmp_recall = tp / (tp + fn)
                
                sum_loss += tmp_loss
                sum_iou += tmp_iou
                sum_prec += tmp_prec
                sum_acc += tmp_acc
                sum_recall += tmp_recall
                
            sum_loss /= c
            sum_acc /= (np.sum(cfsmat)) 
            sum_prec /= c
            sum_recall /= c
            sum_iou /= c
            sum_f1 = 2 * sum_prec * sum_recall / (sum_prec + sum_recall)
            
            loss_score.append(sum_loss)
            iou_score.append(sum_iou)
            acc_score.append(sum_acc)
            pre_score.append(sum_prec)
            recall_score.append(sum_recall)
            f1_score.append(sum_f1)
            torch.cuda.empty_cache()
            
    time_elapsed = time.time() - since
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(time_cost)/len(time_cost))))
    print('mean Loss:',np.mean(loss_score),np.std(loss_score))
    print('mean IoU:',np.mean(iou_score),np.std(iou_score))
    print('mean accuracy:',np.mean(acc_score),np.std(acc_score))
    print('mean precsion:',np.mean(pre_score),np.std(pre_score))
    print('mean recall:',np.mean(recall_score),np.std(recall_score))
    print('mean F1-score:',np.mean(f1_score),np.std(f1_score))
    
    output_file = f'result/{args.result_name}/evaluation_results.txt'  # 指定要写入的文件名

    with open(output_file, 'w') as file:
        file.write('Evaluation complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        file.write('FPS: {:.2f}\n'.format(1.0/(sum(time_cost)/len(time_cost))))
        file.write('mean Loss: {} {}\n'.format(np.mean(loss_score), np.std(loss_score)))
        file.write('mean IoU: {} {}\n'.format(np.mean(iou_score), np.std(iou_score)))
        file.write('mean accuracy: {} {}\n'.format(np.mean(acc_score), np.std(acc_score)))
        file.write('mean precision: {} {}\n'.format(np.mean(pre_score), np.std(pre_score)))
        file.write('mean recall: {} {}\n'.format(np.mean(recall_score), np.std(recall_score)))
        file.write('mean F1-score: {} {}\n'.format(np.mean(f1_score), np.std(f1_score)))