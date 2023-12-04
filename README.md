# Pytorch-Implementation-of-Fast-Multiple-Landmark-Localisation-Using-a-Patch-based-Iterative-Network
*Code Writer: Chaeeun Ryu*

**Not official!** <br>
- Official Code Repository in Tensorflow: https://github.com/yuanwei1989/landmark-detection <br>
- Official paper: https://arxiv.org/pdf/1806.06987.pdf 
- Paper arxiv: https://arxiv.org/abs/1806.06987v2


**Note:** 
1. Unlike the original paper, this implementation jointly trains autoencoder with PIN. Therefore, the autoencoder is used instead of the shape model. (The compression of landmarks representation is only enabled if the number of landmarks exceeds 3.)
2. I have referred to the codes from [the original repository](https://github.com/yuanwei1989/landmark-detection), so there are a lot of codes in common. I just made the code in Pytorch & runnable with the latest version of libraries needed.
(+3. Could be minor errors :upside_down_face: :sweat_smile:)

- To-dos:
    - Finish infer.py
    - scheduler added (Not in the original paper)
    - autoencoder enabled (Not in the original paper)
    - augmentation added (Not in the original paper)