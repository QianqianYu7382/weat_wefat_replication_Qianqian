import logging as log
import math
import itertools as it
import random

import numpy as np
from numpy import dot
from numpy.linalg import norm
import scipy.special
import scipy.stats
from statistics import NormalDist



def unit_vector(vec):
    """
    Returns unit vector
    """
    return vec / np.linalg.norm(vec)


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)
    """
    dotProduct = 0.0
    magnitude1 = 0.0
    magnitude2 = 0.0

    for i in range(len(v1)):
        dotProduct += v1[i] * v2[i]
        magnitude1 += v1[i] * v1[i]
        magnitude2 += v2[i] * v2[i]
    magnitude1 = math.sqrt(magnitude1)
    magnitude2 = math.sqrt(magnitude2)
    if magnitude1 != 0 or magnitude2 != 0:
        cosineSimilarity = dotProduct /(magnitude2*magnitude1)
    else:
        return 0.0;
    return cosineSimilarity

def weat_association(W, A, B):
    """
    Returns association of the word w in W with the attribute for WEAT score.
    s(w, A, B)
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: (len(W), ) shaped numpy ndarray. each rows represent association of the word w in W
    """

    ##return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)
    sum_w_ab = 0
    distribution = []
    for w in W:
        sum_cos_w_a = 0
        sum_cos_w_b = 0
        for a in A:
            cos_w_a = cos_sim(w,a)
            sum_cos_w_a += cos_w_a
        mean_cos_w_a = sum_cos_w_a/len(A)
        for b in B:
            cos_w_b = cos_sim(w,b)
            sum_cos_w_b += cos_w_b
        mean_cos_w_b = sum_cos_w_b/len(B)
        distribution.append(mean_cos_w_a-mean_cos_w_b)
        sum_w_ab = sum_w_ab + mean_cos_w_a-mean_cos_w_b
    W_AB = sum_w_ab/len(W)
    return W_AB, distribution


def weat_differential_association(X, Y, A, B):
    """
    Returns differential association of two sets of target words with the attribute for WEAT score.
    s(X, Y, A, B)
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: differential association (float value)
    """
    X_AB, distributionX = weat_association(X, A, B)
    Y_AB, distributionY = weat_association(Y, A, B)
    return X_AB-Y_AB

def get_both(X,Y):
    both = []
    for x in X:
        both.append(x)
    for y in Y:
        both.append(y)
    return both


def getEntireDistribution(X,Y,A,B):
    both = get_both(X,Y)
    print("Getting the entire distribution...")
    both_AB, distribution = weat_association(both,A,B)
    #print("len of both_dis")
    #print(len(distribution))
    return distribution

def nullDistribution(X,Y,A,B,iteration):
    ## permute concepts and for each permutation calculate getTestStatistic and save it in your distribution
    both = get_both(X,Y)
    ##print("len X Y",X[0],Y[0],"null len X+Y", both[0])
    width, height = len(both), len(A)
    A_null_Matrix = [[0 for x in range(width)] for y in range(height)]
    B_null_Matrix = [[0 for x in range(width)] for y in range(height)]
    for i in range(len(A)):
        a = A[i]
        for j in range(len(both)):
            concept = both[j]
            cos_null_a = cos_sim(concept, a)
            A_null_Matrix[i][j] = cos_null_a
    for k in range(len(B)):
        b = B[k]
        for n in range(len(both)):
            concept1 = both[n]
            cos_null_b = cos_sim(concept1, b)
            B_null_Matrix[k][n] = cos_null_b

    ## assuming that both concepts have the same number of elements
    target_size = len(both)/2
    print("Number of permutations:")
    print(iteration)
    distribution = []
    shuffle = []
    for num in range(len(both)):
        shuffle.append(num)

    for iter in range(iteration):
        random.shuffle(shuffle)

        ## calcualte mean for each null shuffle

        mean_cos_xa = sum_cos_conTobis(target_size,A,A_null_Matrix,shuffle,"x")/(len(A)*target_size)
        mean_cos_xb = sum_cos_conTobis(target_size,B,B_null_Matrix,shuffle,"x")/(len(B)*target_size)
        mean_cos_ya = sum_cos_conTobis(target_size,A,A_null_Matrix,shuffle,"y")/(len(A)*target_size)
        mean_cos_yb = sum_cos_conTobis(target_size,B,B_null_Matrix,shuffle,"y")/(len(B)*target_size)
        distribution.append((mean_cos_xa-mean_cos_xb)-mean_cos_ya+mean_cos_yb)
    return distribution

def sum_cos_conTobis(target_size, A, null_Matrix,shuffle_index,con_type):
    sum_conTobias = 0
    if con_type == "x":
        for i in range(len(A)):
            for j in range(int(target_size)):
                sum_conTobias = sum_conTobias + null_Matrix[i][shuffle_index[j]]
    else:
        for i in range(len(A)):
            for j in range(int(target_size)):
                sum_conTobias = sum_conTobias + null_Matrix[i][shuffle_index[j+int(target_size)]]
    return sum_conTobias


def find_std(distribution):
    distribution.sort()
    mean = np.mean(distribution)
    sqr_sum = 0
    for i in range(len(distribution)):
        sqr_sum += math.pow(distribution[i]-mean,2)
    return math.sqrt((sqr_sum)/(len(distribution)-1))

def effectSize(distribution,difference):
    return difference/find_std(distribution)

def calculateCumulativeProbability(distribution, difference, distribution_type):
    cumulative = -100
    if distribution_type == "empirical" :
        distribution.sort()
        ## empirical distribution
    if distribution_type == "normal":
        distribution.sort()
        cumulative = NormalDist(mu=np.mean(distribution), sigma=find_std(distribution)).cdf(difference)
    return cumulative



def get_p_value(X,Y,A,B,distribution_type,iteration):
    ##P_vec[p_value,effectSize(WEAT_score),std(null_dis)]
    p_value = [0,0,0,0]
    difference = weat_differential_association(X,Y,A,B)
    null_distribution = nullDistribution(X,Y,A,B,iteration)
    entire_distribution = getEntireDistribution(X,Y,A,B)
    p_value[0] = 1- calculateCumulativeProbability(null_distribution,difference,distribution_type)
    p_value[1] = effectSize(entire_distribution,difference)
    p_value[2] = find_std(null_distribution)
    p_value[3] = weat_differential_association(X,Y,A,B)
    return p_value
