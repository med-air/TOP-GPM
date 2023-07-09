from models import *
import os
import glob
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from idhp_data import *
import SimpleITK as sitk
import cv2
import numpy as np
import math



def policy_val(t, yf, q_t0, q_t1, compute_policy_curve=False):
    # if np.any(np.isnan(eff_pred)):
        # return np.nan, np.nan
    q_cat = np.concatenate((q_t0, q_t1),1)

    policy = np.argmax(q_cat,1)
    policy = policy[:,np.newaxis]
    t0_overlap = (policy==t)*(t==0)
    t1_overlap = (policy==t)*(t==1)
    
    
    if np.sum(t0_overlap) == 0:
        t0_value = 0
    else: 
        t0_value = np.mean(yf[t0_overlap])
        
    if np.sum(t1_overlap) == 0:
        t1_value = 0
    else: 
        t1_value = np.mean(yf[t1_overlap])
        
    
    
    pit_0 = np.sum(policy==0)/len(t)
    pit_1 = np.sum(policy==1)/len(t)

    policy_value = pit_0*t0_value + pit_1*t1_value 

    
    return policy_value

def factual_acc(t, yf, q_t0, q_t1):

    q_t0[q_t0>=0.5] = 1
    q_t0[q_t0<0.5] = 0
    
    q_t1[q_t1>=0.5] = 1
    q_t1[q_t1<0.5] = 0

    
    accuracy_0 = np.sum(q_t0[t==0]==yf[t==0])/len(yf[t==0])
    accuracy_1 = np.sum(q_t1[t==1]==yf[t==1])/len(yf[t==1])

    
    print("Factual accuracy of t0:", accuracy_0)
    print("Factual accuracy of t1:", accuracy_1)
    
    return accuracy_0,accuracy_1

def factual_auc(t, yf, q_t0, q_t1):
    from sklearn import metrics
    y_t0 = []
    y_t1 = []
    p_t0 = []
    p_t1 = []
    
    for index in range(len(t)):
        if t[index] ==0:
            y_t0.append(yf[index])
            p_t0.append(q_t0[index])
        else:
            y_t1.append(yf[index])
            p_t1.append(q_t1[index])
    
    
    y_t0,p_t0, y_t1,p_t1 = np.array(y_t0), np.array(p_t0), np.array(y_t1), np.array(p_t1)
    auc0 = metrics.roc_auc_score(y_t0,p_t0)
    auc1 = metrics.roc_auc_score(y_t1,p_t1)
    

    
    print("Factual auc of t0:", auc0)
    print("Factual auc of t1:", auc1)
    
    return auc0,auc1
    
def policy_risk_multi(t, yf, q_t0, q_t1):
    policy_value = policy_val(t, yf, q_t0, q_t1)
    policy_risk = 1 - policy_value
    return policy_risk
  
def ate_error_0_1(t, yf, eff_pred):
    att = np.mean(yf[t==0]) - np.mean(yf[t==1])
    pred_att = np.mean(eff_pred)
    
    return np.abs(att-pred_att)
    
def ate_error_0_2(t, yf, eff_pred):
    att = np.mean(yf[t==0]) - np.mean(yf[t==2])
    pred_att = np.mean(eff_pred)
    
    return np.abs(att-pred_att)
    
def ate_error_1_2(t, yf, eff_pred):
    att = np.mean(yf[t==1]) - np.mean(yf[t==2])
    pred_att = np.mean(eff_pred)
    
    return np.abs(att-pred_att)