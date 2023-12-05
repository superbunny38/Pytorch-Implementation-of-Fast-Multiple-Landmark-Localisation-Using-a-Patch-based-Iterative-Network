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

    return landmarks