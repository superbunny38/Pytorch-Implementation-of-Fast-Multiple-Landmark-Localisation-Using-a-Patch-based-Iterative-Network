#update_landmarks(landmarks, action_prob, yr_val, config)#update rule A, B, C

import numpy as np

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