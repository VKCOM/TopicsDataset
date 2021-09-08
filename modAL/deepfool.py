from typing import Union, Tuple

from modAL.utils import multi_argmax

from modAL.models.base import BaseLearner, BaseCommittee

import scipy.sparse as sp

import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients

IMG_LEN = 1024
TXT_LEN = 300
N_CLASSES = 50


# original: https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py
def deepfool(image, txt, net, num_classes=N_CLASSES, overshoot=0.02, max_iter=50):
    """
       :param image: Image embedding of size 1024
       :param txt: Text embedding of size 300
       :param net: network (input: images, texts, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 50)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()
        txt = txt.cuda()
        net = net.cuda()

    f_img_txt = net.forward(
        Variable(image, requires_grad=True),
        Variable(txt, requires_grad=True)
    ).data.cpu().numpy().flatten()

    I = (np.array(f_img_txt)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_img_shape = image.cpu().numpy().shape
    input_txt_shape = txt.cpu().numpy().shape

    pert_img = copy.deepcopy(image)
    pert_txt = copy.deepcopy(txt)

    w_img = np.zeros(input_img_shape)
    w_txt = np.zeros(input_txt_shape)

    r_img_tot = np.zeros(input_img_shape)
    r_txt_tot = np.zeros(input_txt_shape)

    loop_i = 0

    # print('pert img shape:', pert_img.shape)
    # print('pert txt shape:', pert_txt.shape)

    x_img = Variable(pert_img, requires_grad=True)
    x_txt = Variable(pert_txt, requires_grad=True)

    fs = net.forward(x_img, x_txt)

    k_i = label

    while k_i == label and loop_i < max_iter:

        pert_img_g = np.inf
        pert_txt_g = np.inf

        fs[0, I[0]].backward(retain_graph=True)
        grad_img_orig = x_img.grad.data.cpu().numpy().copy()
        grad_txt_orig = x_txt.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x_img)
            zero_gradients(x_txt)

            fs[0, I[k]].backward(retain_graph=True)
            cur_img_grad = x_img.grad.data.cpu().numpy().copy()
            cur_txt_grad = x_txt.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_img_k = cur_img_grad - grad_img_orig
            w_txt_k = cur_txt_grad - grad_txt_orig

            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_img_k = abs(f_k) / np.linalg.norm(w_img_k.flatten())
            pert_txt_k = abs(f_k) / np.linalg.norm(w_txt_k.flatten())

            # determine which w_k to use
            if pert_img_k < pert_img_g:
                pert_img_g = pert_img_k
                w_img = w_img_k

            if pert_txt_k < pert_txt_g:
                pert_txt_g = pert_txt_k
                w_txt = w_txt_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_img_i = (pert_img_g + 1e-4) * w_img / np.linalg.norm(w_img)
        r_txt_i = (pert_txt_g + 1e-4) * w_txt / np.linalg.norm(w_txt)

        r_img_tot = np.float32(r_img_tot + r_img_i)
        r_txt_tot = np.float32(r_txt_tot + r_txt_i)

        if is_cuda:
            pert_img = image + (1 + overshoot) * torch.from_numpy(r_img_tot).cuda()
            pert_txt = txt + (1 + overshoot) * torch.from_numpy(r_txt_tot).cuda()
        else:
            pert_img = image + (1 + overshoot) * torch.from_numpy(r_img_tot)
            pert_txt = txt + (1 + overshoot) * torch.from_numpy(r_txt_tot)

        x_img = Variable(pert_img, requires_grad=True)
        x_txt = Variable(pert_txt, requires_grad=True)

        fs = net.forward(x_img, x_txt)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_img_tot = (1 + overshoot) * r_img_tot
    r_txt_tot = (1 + overshoot) * r_txt_tot

    return r_img_tot, r_txt_tot, loop_i, label, k_i, pert_img, pert_txt


def deepfool_sampling(
    classifier: Union[BaseLearner, BaseCommittee],
    X: Union[np.ndarray, sp.csr_matrix],
    n_instances: int = 20,
) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix, list]]:

    model = classifier.estimator.model

    perts = [
        deepfool(
            torch.tensor([X[0][i]]).float(),
            torch.tensor([X[1][i]]).float(),
            model
        ) for i in range(len(X[0]))
    ]

    perts_metrics = np.array([np.linalg.norm(pert[0]) + np.linalg.norm(pert[1]) for pert in perts])

    # more perturbation => more stable object
    query_idx = multi_argmax(1 - perts_metrics, n_instances=n_instances)

    if isinstance(X, list) and isinstance(X[0], np.ndarray):
        new_batch = [x[query_idx] for x in X]
    else:
        new_batch = X[query_idx]

    return query_idx, new_batch
