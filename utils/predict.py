#update_landmarks(landmarks, action_prob, yr_val, config)#update rule A, B, C

import numpy as np
from . import patch
import torch
import torch.nn as nn
import scipy.ndimage


def get_p_max(action_prob,config):
    """Get P_max for Rule C

    Args:
        action_prob: Prob (cls output before softmax)

    Returns:
        P_max
    """
    p_max = []
    for candi in action_prob:#action_prob: (6,12) == [n_candidates, cls_output]
        p_max_candi = []
        for landmark_idx in range(config.landmark_count):
            tmp_p_max = []
            for dir_idx in range(0,6,2):
                tmp_p_max.append(max(candi[(landmark_idx+1)*dir_idx],candi[(landmark_idx+1)*dir_idx+1]))
            p_max_candi.append(tmp_p_max)
        p_max.append(p_max_candi)
    return np.array(p_max)

def reshape_yr_val(yr_val):
    reshaped_yr_val = []
    for candi in yr_val:
        reshaped_yr_val.append(np.array([candi[:3],candi[3:]]))
    return np.array(reshaped_yr_val)

def update_landmarks(landmarks, action_prob, yr_val, config):#update rule A,B,C
    """PIN update rule A,B,C
    Update x_t -> x_{t-1}
    
    Args:
        landmarks (_type_): _description_
        action_prob (_type_): _description_
        yr_val (_type_): _description_
        config (_type_): _description_
    """
    if config.predict_mode == 0:
        ind = np.argmax(np.amax(np.reshape(action_prob, (b.shape[0], b.shape[1], 2)), axis = 2), axis = 1)#ind = [num_examples]
        row_ind = np.arange(landmarks.shape[0])
        landmarks[row_ind,ind] = landmarks[row_ind,ind] - yr_val[row_ind,ind]

    elif config.predict_mode == 1:
        # print("lanmarks shape: ",landmarks.shape)
        # print("yr_val shape: ",yr_val.shape)
        
        updated_landmarks = landmarks.reshape(config.num_random_init+1,-1) - yr_val * np.amax(np.reshape(action_prob, (landmarks.reshape(config.num_random_init+1,-1).shape[0], landmarks.reshape(config.num_random_init+1,-1).shape[1], 2)), axis=2)
        landmarks = updated_landmarks.reshape(landmarks.shape)
        
    elif config.predict_mode == 4:#coded by me! (inefficient)
        # print("landmarks shape: ",landmarks.shape) landmarks shape:  (6, 2, 3)
        # print("action_prob shape: ",action_prob.shape) action_prob shape:  (6, 12)
        # print("yr_val shape: ",yr_val.shape) yr_val shape:  (6, 6) == d
        weighted_move = get_p_max(action_prob,config)
        # print("weighted_move shape: ", weighted_move.shape)
        landmarks = landmarks - reshape_yr_val(yr_val)*weighted_move
        # landmarks = landmarks - yr_val*np.amax(np.reshape(action_prob, (landmarks.shape[0], landmarks.shape[1],2)), axis =2)

    elif config.predict_mode == 2:
        landmarks = landmarks - yr_val

    elif config.predict_mode == 3:
        step = 1
        action_prob_reshape = np.reshape(action_prob, (landmarks.shape[0], landmarks.shape[1], 2))
        ind = np.argmax(np.amax(action_prob_reshape, axis = 2), axis = 1)#ind = [num_exmples]
        row_ind = np.arange(landmarks.shape[0])
        is_negative = np.argmax(action_prob_reshape[row_ind,ind],axis = 1)
        landmarks[row_ind[is_negative],ind[is_negative]] = landmarks[row_ind[is_negative],ind[is_negative]] + step
        landmarks[row_ind[np.logical_not(is_negative)],ind[np.logical_not(is_negative)]] = landmarks[row_ind[np.logical_not(is_negative)],ind[np.logical_not(is_negative)]] - step

    return landmarks


def predict_cnn(iter_idx, patches, yc_gt, yr_gt, model, config, is_softmax = True):
    """Shape analysis

    yr_val: (5,6)
    """
    model.eval()
    patches_eval = torch.from_numpy(patches).float().to(config.device)
    patches_eval = patches_eval.permute(0, 3, 1, 2)
    
    yc_val,yr_val = model(patches_eval)
    
    if is_softmax:
        yc_val = nn.Softmax(dim = 1)(yc_val)
    
    yc_val,yr_val = yc_val.detach().cpu().numpy(), yr_val.detach().cpu().numpy()
    
    
    
    # Compute classification probabilities
    action_prob = np.exp(yc_val - np.expand_dims(np.amax(yc_val, axis=1), 1))
    action_prob = action_prob / np.expand_dims(np.sum(action_prob, axis=1), 1)  # action_prob=[num_examples, 2*num_shape_params]
    
    
    return yc_val, yr_val, action_prob, yc_gt, yr_gt

def predict_landmarks(dataset, config, model, train = True):
    """Shape analysis
    img: (324, 207, 279, 1)
    label: (2, 3)
    bs_gt: (1,6)
    landmarks: (2,3)
    dbs: (5,6)
    bs: (5,6)
    hat_landmarks: (5,6)
    patches[0]: (101,101,6)
    """
    
    def get_train_pairs(image,label,config):
        bs_gt = label.reshape(-1,config.landmark_count*3)
        landmarks = label
        num_actions = 6#axis가 3개라서 3x2로 6으로 고정
        actions_ind = np.zeros((config.num_random_init), dtype = np.float32)
        actions = np.zeros((config.num_random_init, num_actions), dtype = np.float32)
        
        #Randomly sample x (Random initialization)
        #Randomly sampled init points: bs
        bounds = config.sd*np.sqrt(config.landmark_count)
        bs = np.random.rand(config.num_random_init, config.num_cnn_output_r)*2*bounds-bounds
        dbs = bs - bs_gt#What regressor should predict
        
        #Get classification label
        actions_ind = np.zeros((config.num_random_init), dtype=np.float32)
        actions = np.zeros((config.num_random_init, num_actions), np.float32)
        max_db_ind = np.argmax(np.abs(dbs), axis = 1)
        max_db = dbs[np.arange(dbs.shape[0]), max_db_ind]
        is_positive = (max_db > 0.5)
        actions_ind[is_positive] = max_db_ind[is_positive]*2
        actions_ind[np.logical_not(is_positive)] = max_db_ind[np.logical_not(is_positive)]*2 + 1
        actions_ind = actions_ind.astype(int)
        actions[np.ix_(np.arange(config.num_random_init), actions_ind)] = 1
        return actions, dbs, bs
    
    num_landmarks = config.landmark_count
    max_test_steps = config.max_test_steps
    patch_size = config.patch_size
    patch_r = int((patch_size - 1) / 2)
    img_count = dataset.__len__()
    
    landmark_all_steps_ret = np.zeros((img_count, max_test_steps+1,config.num_random_init, num_landmarks, 3))
    final_landmarks_ret = np.zeros((img_count, num_landmarks, 3))
    if train:
        for img_idx in range(dataset.__len__()):
            img, label = dataset.__getitem__(img_idx)
            actions, dbs, hat_landmarks = get_train_pairs(img, label, config)
            patches = np.zeros((config.num_random_init, config.patch_size, config.patch_size, config.landmark_count*3))
            
            for candidate_idx in range(config.num_random_init):
                patches[candidate_idx] = patch.extract_patch_all_landmarks(img, hat_landmarks.reshape(config.num_random_init,config.landmark_count,3)[candidate_idx], patch_r)
                
            landmark_all_steps = np.zeros((config.max_test_steps+1, config.num_random_init, config.landmark_count,3))
            landmark_all_steps[0] = hat_landmarks.reshape(config.num_random_init,config.landmark_count,3)
            
            
            for iter_idx in range(config.max_test_steps):
                yc_val, yr_val, action_prob, yc_gt, yr_gt = predict_cnn(iter_idx, patches, actions, dbs, model, config)
                #update predicted landmarks
                hat_landmarks = hat_landmarks - yr_val*np.amax(np.reshape(action_prob,(hat_landmarks.shape[0], hat_landmarks.shape[1],2)),axis = 2)
                
                # weighted_move = get_p_max(action_prob,config)
                # # print("weighted_move shape: ", weighted_move.shape)
                # hat_landmarks = hat_landmarks - (reshape_yr_val(yr_val)*weighted_move).reshape(config.num_random_init,-1)
                
                landmark_all_steps[iter_idx+1] = hat_landmarks.reshape(config.num_random_init,config.landmark_count,3)
                
                for candi_idx in range(config.num_random_init):
                    patches[candi_idx] = patch.extract_patch_all_landmarks(img, hat_landmarks.reshape(config.num_random_init,config.landmark_count,3)[candi_idx], patch_r)

                print(f"====iteration {iter_idx}=====")
                print(np.mean(hat_landmarks, axis = 0).reshape(config.landmark_count, -1))
            # print("landmark_all_steps shape:")
            # print(landmark_all_steps.shape)
            # print("hat_landmarks shape:")
            # print(hat_landmarks.shape)
            final_landmarks = np.mean(hat_landmarks, axis = 0).reshape(config.landmark_count, -1)
            # print("final_landmarks shape:", final_landmarks.shape)
            # final_landmarks = np.mean(landmark_all_steps[-1,:,:,:], axis = 0)
            final_landmarks_ret[img_idx] = final_landmarks
            landmark_all_steps_ret[img_idx] = landmark_all_steps
            
        return final_landmarks_ret, landmark_all_steps_ret