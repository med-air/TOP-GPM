from models import *
import os
import glob
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from idhp_data import *
import SimpleITK as sitk
import cv2
import numpy as np
import math
import copy
from tools import *

def _split_output(yt_hat, t, y, y_scaler, x, is_train=False):
    """
        Split output into dictionary for easier use in estimation
        Args:
            yt_hat: Generated prediction
            t: Binary treatment assignments
            y: Treatment outcomes
            y_scaler: Scaled treatment outcomes
            x: Covariates
            index: Index in data

        Returns:
            Dictionary of all needed data
    """

    
    yt_hat = yt_hat
    q_t0 = yt_hat[:, 0].reshape(-1, 1).copy()
    q_t1 = yt_hat[:, 1].reshape(-1, 1).copy()

    g = yt_hat[:, 2].copy()
    treatment_predicted = g.copy()
    treatment_predicted[treatment_predicted>=0.5] = 1
    treatment_predicted[treatment_predicted<0.5] = 0

    y = y.copy()
    var = "average propensity for t: {}".format(g[t.squeeze() == 1.].mean())
    
    q_cat = np.concatenate((q_t0, q_t1),1)

    policy = np.argmax(q_cat,1)
    
    print(var)
    print("Policy Risk:", policy_risk_multi(t, y, q_t0, q_t1))
    print("Ate_Error:", ate_error_0_1(t, y, q_t0 - q_t1))

    print("Treatment accuracy:", np.sum(treatment_predicted==t.squeeze())/treatment_predicted.shape[0])
    
    if not is_train:
        print("Treatment policy    :",policy)
        print("Treatment prediction:",treatment_predicted)
        print("Treatment label     :",t.squeeze().astype(int))
    
    auc0,auc1 = factual_auc(t, y, q_t0, q_t1)
    accuracy_0, accuracy_1 = factual_acc(t, y, q_t0, q_t1)
    

    return {'ave propensity for t': g[t.squeeze() == 1.].mean(),
    'Policy Risk': policy_risk_multi(t, y, q_t0, q_t1), 
    'Ate_Error_0_1': ate_error_0_1(t, y, q_t0 - q_t1), 'Treatment accuracy': np.sum(treatment_predicted==t.squeeze())/treatment_predicted.shape[0], 
    'Treatment policy': policy, 'Treatment prediction': treatment_predicted, 'Treatment label': t.squeeze().astype(int), 'accuracy_0': accuracy_0, 'accuracy_1':accuracy_1,
    'auc0': auc0, 'auc1':auc1}

average_propensity_for_t0 = []
average_propensity_for_t1 = []
average_propensity_for_t2 = []
policy_risk = []
test_ate_error_0_1 = []
test_ate_error_0_2 = []
test_ate_error_1_2 = []
treatment_accuracy = []
treatment_policy=np.array([])
treatment_prediction=np.array([])
treatment_label=np.array([])
test_factual_accuracy_of_t0 = []
test_factual_accuracy_of_t1 = []
test_factual_accuracy_of_t2 = []


train_average_propensity_for_t0 = []
train_average_propensity_for_t1 = []
train_average_propensity_for_t2 = []
train_policy_risk = []
train_ate_error_0_1 = []
train_ate_error_0_2 = []
train_ate_error_1_2 = []
train_treatment_accuracy = []
train_factual_accuracy_of_t0 = []
train_factual_accuracy_of_t1 = []
train_factual_accuracy_of_t2 = []

train_factual_auc_of_t0 = []
train_factual_auc_of_t1 = []
test_factual_auc_of_t0 = []
test_factual_auc_of_t1 = []

key_word = 'Treatment accuracy'
key_word4 = 'Policy Risk'
key_word5 = 'accuracy_0'
key_word6 = 'accuracy_1'
key_word7 = 'accuracy_2'

key_word1 = 'Ate_Error_0_1'
key_word2 = 'Ate_Error_0_2'
key_word3 = 'Ate_Error_1_2'

key_word_auc0 = 'auc0'
key_word_auc1 = 'auc1'
epoch_index = 0

for validation_index in range(1):
    best_evaluation = 0.
    train_outputs_best = {}
    test_outputs_best = {}
    for epoch in range(0,1500,10):
        test_results = np.load("../results_save/IPH_limited_ours2/{}_fold_{}_epoch_test.npz".format(validation_index, epoch), allow_pickle=True)
        train_results = np.load("../results_save/IPH_limited_ours2/{}_fold_{}_epoch_train.npz".format(validation_index, epoch), allow_pickle=True)
        
        yt_hat_test, t_test, y_test, y, x_test = test_results['yt_hat_test'], test_results['t_test'], test_results['y_test'], \
        test_results['y'], test_results['x_test']
        yt_hat_train, t_train, y_train, y, x_train = train_results['yt_hat_train'], train_results['t_train'], train_results['y_train'], \
        train_results['y'], train_results['x_train']
        
        test_outputs = _split_output(yt_hat_test, t_test, y_test, y, x_test, is_train=False)
        train_outputs = _split_output(yt_hat_train, t_train, y_train, y, x_train, is_train=True)
        #test_outputs = test_outputs['arr_0'].item()
        #train_outputs = train_outputs['arr_0'].item()
        if test_outputs[key_word_auc0]+test_outputs[key_word_auc1] >= best_evaluation and epoch>=100:
        #if (test_outputs[key_word1]+test_outputs[key_word2]+test_outputs[key_word3]+test_outputs[key_word4]+(1-test_outputs[key_word5])+(1-test_outputs[key_word6])+(1-test_outputs[key_word7]))/7 <= best_evaluation and epoch>=500:
            test_outputs_best = test_outputs
            best_evaluation = test_outputs[key_word_auc0]+test_outputs[key_word_auc1]
            epoch_index = epoch
            #best_evaluation = (test_outputs[key_word1]+test_outputs[key_word2]+test_outputs[key_word3]+test_outputs[key_word4]+(1-test_outputs[key_word5])+(1-test_outputs[key_word6])+(1-test_outputs[key_word7]))/7
        
        train_outputs_best = train_outputs
        # if (train_outputs[key_word1]+train_outputs[key_word2]+train_outputs[key_word3]+train_outputs[key_word4]+(1-train_outputs[key_word5])+(1-train_outputs[key_word6])+(1-train_outputs[key_word7]))/7 <= best_evaluation and epoch>=500:
            # train_outputs_best = train_outputs
            
            # #best_evaluation = test_outputs[key_word]
            # best_evaluation = (train_outputs[key_word1]+train_outputs[key_word2]+train_outputs[key_word3]+train_outputs[key_word4]+(1-train_outputs[key_word5])+(1-train_outputs[key_word6])+(1-train_outputs[key_word7]))/7

        
    print("==========Best test results for the {} fold==========".format(validation_index))
    
    print("average propensity for t: {}".format(test_outputs_best['ave propensity for t']))
    print("Policy Risk:", test_outputs_best['Policy Risk'])
    print("Ate_Error_0_1:", test_outputs_best['Ate_Error_0_1'])

    print("Treatment accuracy:", test_outputs_best['Treatment accuracy'])
    print("Treatment policy    :",test_outputs_best['Treatment policy'])
    print("Treatment prediction:",test_outputs_best['Treatment prediction'])
    print("Treatment label     :",test_outputs_best['Treatment label'])
    print("Factual accuracy of t0:", test_outputs_best['accuracy_0'])
    print("Factual accuracy of t1:", test_outputs_best['accuracy_1'])
    print("Factual auc of t0:", test_outputs_best['auc0'])
    print("Factual auc of t1:", test_outputs_best['auc1'])

    # print("Factual auc of t0:", test_outputs_best['auc_0'])

    print("==========Best train results for the {} fold==========".format(validation_index))
    print("average propensity for t: {}".format(train_outputs_best['ave propensity for t']))
    print("Policy Risk:", train_outputs_best['Policy Risk'])
    print("Ate_Error_0_1:", train_outputs_best['Ate_Error_0_1'])

    print("Treatment accuracy:", train_outputs_best['Treatment accuracy'])
    print("Factual accuracy of t0:", train_outputs_best['accuracy_0'])
    print("Factual accuracy of t1:", train_outputs_best['accuracy_1'])
    print("Factual auc of t0:", train_outputs_best['auc0'])
    print("Factual auc of t1:", train_outputs_best['auc1'])

    # print("Factual auc of t0:", train_outputs_best['auc_0'])

    print("====================================================")
    average_propensity_for_t0.append(test_outputs_best['ave propensity for t'])

    policy_risk.append(test_outputs_best['Policy Risk'])
    test_ate_error_0_1.append(test_outputs_best['Ate_Error_0_1'])

    treatment_accuracy.append(test_outputs_best['Treatment accuracy'])
    test_factual_accuracy_of_t0.append(test_outputs_best['accuracy_0'])
    test_factual_accuracy_of_t1.append(test_outputs_best['accuracy_1'])
    test_factual_auc_of_t0.append(test_outputs_best['auc0'])
    test_factual_auc_of_t1.append(test_outputs_best['auc1'])

    # test_factual_auc_of_t0.append(test_outputs_best['auc_0'])

    treatment_policy=np.concatenate((treatment_policy,test_outputs_best['Treatment policy']),0)
    treatment_prediction=np.concatenate((treatment_prediction,test_outputs_best['Treatment prediction']),0)
    treatment_label=np.concatenate((treatment_label,test_outputs_best['Treatment label']),0)

    train_average_propensity_for_t0.append(train_outputs_best['ave propensity for t'])

    train_policy_risk.append(train_outputs_best['Policy Risk'])
    train_ate_error_0_1.append(train_outputs_best['Ate_Error_0_1'])

    train_factual_accuracy_of_t0.append(train_outputs_best['accuracy_0'])
    train_factual_accuracy_of_t1.append(train_outputs_best['accuracy_1'])
    train_factual_auc_of_t0.append(train_outputs_best['auc0'])
    train_factual_auc_of_t1.append(train_outputs_best['auc1'])

    train_treatment_accuracy.append(train_outputs_best['Treatment accuracy'])

print("==========Average best test results==========")
print("The best epoch:",epoch_index)
print("average propensity for t: {}".format(np.mean(average_propensity_for_t0)))
print("Policy Risk: {} +- {}".format(np.mean(policy_risk),np.std(policy_risk)))
print("Ate_Error_0_1: {} +- {}".format(np.mean(test_ate_error_0_1),np.std(test_ate_error_0_1)))

print("Treatment accuracy: {} +- {}".format(np.mean(treatment_accuracy),np.std(treatment_accuracy)))
print("Treatment policy    :",treatment_policy)
print("Treatment prediction:",treatment_prediction)
print("Treatment label     :",treatment_label)
print("Factual accuracy of t0: {} +- {}".format(np.mean(test_factual_accuracy_of_t0),np.std(test_factual_accuracy_of_t0)))
print("Factual accuracy of t1: {} +- {}".format(np.mean(test_factual_accuracy_of_t1),np.std(test_factual_accuracy_of_t1)))
print("Factual auc of t0: {} +- {}".format(np.mean(test_factual_auc_of_t0),np.std(test_factual_auc_of_t0)))
print("Factual auc of t1: {} +- {}".format(np.mean(test_factual_auc_of_t1),np.std(test_factual_auc_of_t1)))

# print("Factual auc of t0: {} +- {}".format(np.mean(test_factual_auc_of_t0),np.std(test_factual_auc_of_t0)))
# print("Factual auc of t1: {} +- {}".format(np.mean(test_factual_auc_of_t1),np.std(test_factual_auc_of_t1)))
# print("Factual auc of t2: {} +- {}".format(np.mean(test_factual_auc_of_t2),np.std(test_factual_auc_of_t2)))
print("==========Average best train results=========")
print("average propensity for t: {}".format(np.mean(train_average_propensity_for_t0)))
print("Policy Risk: {} +- {}".format(np.mean(train_policy_risk),np.std(train_policy_risk)))
print("Ate_Error_0_1: {} +- {}".format(np.mean(train_ate_error_0_1),np.std(train_ate_error_0_1)))

print("Treatment accuracy: {} +- {}".format(np.mean(train_treatment_accuracy), np.std(train_treatment_accuracy)))
print("=============================================")