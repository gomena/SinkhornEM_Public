import numpy as np
from time import time
import scipy.misc
import scipy
from scipy.special import logsumexp
from copy import deepcopy
from scipy.linalg import solve_triangular
import sklearn


def em_algorithm_single(samples, mu0, sigma, log_theta0, n_iter=100, update_theta=True):
    theta0 = np.exp(log_theta0)
    means_em = mu0
    n_samples = samples.shape[0]
    N = n_samples
    d = samples.shape[1]
    seq = []
    thetaseq = []
    for i_ in range(n_iter):
        K = create_K(means_em, samples)
        dif = - K / (2 * sigma)
        dif = dif + np.tile(log_theta0, [1, n_samples])
        weights = np.exp(dif - logsumexp(dif, axis=0, keepdims=True)) / N
        dif = (weights[0, :] - weights[1, :])

        means_em0 = np.sum(dif[:, np.newaxis] * samples, axis=0)
        means_em = np.zeros((2, d))
        means_em[0, 0] = means_em0[0]
        means_em[1, 0] = -means_em0[0]
        seq.append(means_em)
        thetaseq.append(theta0)
        if update_theta is True:
            K = create_K(means_em, samples)
            dif = - K / (2 * sigma) - d * 0.5 * np.log((2 * np.pi * sigma))
            dif = dif + np.tile(log_theta0, [1, n_samples])
            weights = np.exp(dif - logsumexp(dif, axis=0, keepdims=True))
            theta0 = np.mean(weights, 1, keepdims=True)
            log_theta0 = np.log(theta0)

    return means_em, thetaseq, seq


def sinkhorn_em_algorithm_single(samples, mu0, sigma, log_theta0, n_iter=100, n_iter_sinkhorn=100):
    theta0 = np.exp(log_theta0)

    means_em = mu0
    n_samples = samples.shape[0]
    d = samples.shape[1]
    seq = []
    for _ in range(n_iter):
        K = create_K(means_em, samples)
        _, weights, _, _ = sinkhorn_np_logspace(K, 2 * sigma, mu_x=theta0.flatten(),
                                                mu_y=np.ones((n_samples)) / n_samples, n_iter=n_iter_sinkhorn)
        dif = (weights[0, :] - weights[1, :])

        means_em0 = np.sum(dif[:, np.newaxis] * samples, axis=0)
        means_em = np.zeros((2, d))
        means_em[0, 0] = means_em0[0]
        means_em[1, 0] = -means_em0[0]
        seq.append(means_em)

    return means_em, weights, seq


def sinkhorn_np_logspace(K, eps,u=None, v=None, mu_x=None, mu_y=None, n_iter=200):
  if (mu_x is None):
    mu_x = np.ones(K.shape[0]) / K.shape[0]
  if (mu_y is None):
    mu_y = np.ones(K.shape[1]) / K.shape[1]
  if u is None:
    u = np.zeros(len(mu_x))
  if v is None:
    v = np.zeros(len(mu_y))
  u =u- u[-1]
  v = v-v[-1]
  for _ in range(n_iter):
    M = (-K + u[:, np.newaxis] + v[:, np.newaxis].T) / eps
    u = eps * (np.log(mu_x) - logsumexp(M, axis=1)) + u
    M = (-K + u[:, np.newaxis] + v[:, np.newaxis].T) / eps
    v = eps * (np.log(mu_y) - logsumexp(M.T, axis=1)) + v

  u = eps * (np.log(mu_x) - logsumexp(M, axis=1)) + u
  uu = np.exp(u / eps)
  vv = np.exp(v / eps)

  val1 = np.matmul(np.log(uu / mu_x), mu_x)
  val2 = np.matmul(np.log(vv / mu_y), mu_y)
  val = val1 + val2
  pi = np.dot(np.diag(uu), np.dot(np.exp(-K / eps), np.diag(vv)))
  return val, pi,u,v



def error(theta1,theta2, weights):
    n= theta2.shape[0]
    if n==2:
        perms = [[0,1], [1,0]]
    if n==3:
        perms = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1],[2,1,0]]
    errors = np.zeros(len(perms))
    for l in range(len(perms)):
        theta_aux = theta1[perms[l],:]
        errors[l]=np.sum(weights* np.sum((theta_aux -theta2)**2, axis=1))

    return np.min(errors)


def create_K(x_real, y_real):
  N_x = x_real.shape[0]
  N_y = y_real.shape[0]

  normx = np.tile(np.sum(x_real ** 2, 1, keepdims=True),  [1, N_y])
  normy = np.tile(np.sum(y_real **2 , 1, keepdims=True).T, [N_x, 1])

  z = np.matmul(x_real, y_real.T)
  return (normx - 2 * z + normy)