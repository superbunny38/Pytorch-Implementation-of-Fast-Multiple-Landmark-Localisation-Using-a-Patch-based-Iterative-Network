#update_landmarks(landmarks, action_prob, yr_val, config)#update rule A, B, C

import numpy as np

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
        #landmarks = landmarks - yr_val*np.amax(np.reshape(action_prob, (landmarks.shape[0], landmarks.shape[1],2)), axis =2)
        #above code throws an error
        pass
    
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