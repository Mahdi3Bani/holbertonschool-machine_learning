#!/usr/bin/env python3
import numpy as np

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Perform the Baum-Welch algorithm for a hidden Markov model.
   
    Returns The converged Transition, Emission, or None, None on failure
    """
    
    def forward(Observation, Emission, Transition, Initial):
        '''perfrom the forward algorithm for Baum-Welch'''
        T = Observation.shape[0]
        N = Emission.shape[0]
        F = np.zeros((N, T))

        prev = Initial
        for t in range(T):
            F[:, t] = Emission[:, Observation[t]] * np.reshape(prev, (N))
            prev = F[:, t] @ Transition
        P = np.sum(prev)
        return P, F

    def backward(Observation, Emission, Transition, Initial):
        '''perfrom the forward algorithm for Baum-Welch'''
        T = Observation.shape[0]
        N = Emission.shape[0]
        B = np.zeros((N, T))

        B[:, -1] = 1.0
        for t in range(T-2, -1, -1):
            obs = Emission[:, Observation[t+1]]
            for i in range(N):
                B[i, t] = np.sum(B[:, t+1] * Transition[i, :] * obs)
        P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
        return P, B
    
    M = Transition.shape[0]
    T = Observations.shape[0]
    
    for _ in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)
        
        xi = np.zeros((M, M, T-1))
        for t in range(T - 1):
            denominator = (alpha[:, t] @ Transition * Emission[:,
                                            Observations[t+1]].T) @ beta[:, t+1]
            for i in range(M):
                numerator = (alpha[i, t] * Transition[i,
                            :] * Emission[:,Observations[t+1]].T * beta[:, t+1].T)
                xi[i, :, t] = numerator / denominator
        
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi, axis=2) / np.reshape(np.sum(gamma, axis=1), (-1, 1))
        gamma = np.hstack((gamma, np.reshape(np.sum(xi[:, :, T - 2], axis=0), (-1, 1))))

        K = Emission.shape[1]
        denominator = np.sum(gamma, axis=1)
        for k in range(K):
            Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
        Emission = Emission / np.reshape(denominator, (-1, 1))
    
    return Transition, Emission
