# Pytorch-Implementation-of-Fast-Multiple-Landmark-Localisation-Using-a-Patch-based-Iterative-Network
*Code Writer: Chaeeun Ryu*

**Not official!** <br>
- Official Code Repository in Tensorflow: https://github.com/yuanwei1989/landmark-detection <br>
- Official paper: https://arxiv.org/pdf/1806.06987.pdf 
- Paper arxiv: https://arxiv.org/abs/1806.06987v2


**Note:** 
1. Unlike the original paper, this implementation jointly trains autoencoder with PIN. Therefore, the autoencoder is used instead of the shape model. (The compression of landmarks representation is only enabled if the number of landmarks exceeds 3.)<br>
    [Details]:
    ex) number of landmarks = 2
    
    $c \in \mathbb{R}^{12}, c : \mathrm{{} c_{1_{l_1}}^+,c_{1_{l_1}}^-,c_{2_{l_1}}^+,c_{2_{l_1}}^-,c_{3_{l_1}}^+,c_{3_{l_1}}^-,c_{1_{l_2}}^+,c_{1_{l_2}}^-,c_{2_{l_2}}^+,c_{2_{l_2}}^-,c_{3_{l_2}}^+,c_{3_{l_2}}^- \mathrm{}}$

    $d \in \mathbb{R}^5, d: \mathrm{{} {x_{l_1},y_{l_1},z_{l_1},x_{l_2},y_{l_2},z_{l_2}} \mathrm{}}$
    
    
    $l_i : i^{th} landmark$


2. I have referred to the codes from [the original repository](https://github.com/yuanwei1989/landmark-detection), so there are a lot of codes in common. I just made the code in Pytorch & runnable with the latest version of libraries needed.

3. The plot for loss doesn't show the loss at iteration 0

(+4. Could be minor errors :upside_down_face: :sweat_smile:)

- To-dos:
    - Finish infer.py
    - scheduler added (Not in the original paper)
    - autoencoder enabled (Not in the original paper)
    - augmentation added (Not in the original paper)
    - regression loss를 기준으로 training 멈출 시점을 정하는게 더 효과적으로 보임
    - best model saving (Not in the original paper)