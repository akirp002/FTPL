#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cupy as cp
import numpy as np
from cupy import random
import scipy as sc
from scipy import linalg
import matplotlib as plt
import numpy as np
from numpy import genfromtxt
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import invgamma
import matplotlib.pyplot as plt
from sympy import Matrix
from datetime import datetime
import seaborn as sns
import math
import time
ordqz = sc.linalg.ordqz
svd = sc.linalg.svd
import pandas as pd
resh  = cp.reshape
norm = cp.random.standard_normal
zeros = cp.zeros
import scipy as sc
inv =  sc.linalg.inv
resh = np.reshape
m = np.matmul
eig = sc.linalg.eig


# In[2]:


# Initialize Parameters
psi = 1.6**-1;epsilon = 1.2;alpha = .3;sigma = .1;rho = .9;rho_k = .9;
delta = 2.3**-1;n1 = 1.1**-1;n2 = 1.1**-1
phi_b = 1.5;phi_s= 0.1;theta = .016;phi_x = 0.5;phi_pi = 1.5;phi_t =.2;rh0= .7;h_w = .5;h_r = 1-h_w
p_u = .5;p_r = .5;p_s = .5;p_e = .5;p_a = .5; p_r_star = .5;
sig_u =.2;sig_r = .2;sig_s = .2;sig_e = .2;sig_a = .2;sig_r_star = .1


# In[3]:


def set_para():
    para=  np.zeros([30])
    para[0] =psi;
    para[1] = epsilon;
    para[2] = alpha;
    para[3] = sigma;
    para[4] = rho;
    para[5] = rho_k;
    para[6] = delta;
    para[7] = n1;
    para[8] = n2;
    para[9]= phi_b;
    para[10] =phi_s;
    para[11] = theta;
    para[12] = phi_x;
    para[13] = phi_pi;
    para[14] = phi_t;
    para[15] = rh0;
    para[16] = h_w;
    para[17] = p_u;
    para[18] = p_r;
    para[19] = p_s;
    para[20] = p_e;
    para[21] = p_a;
    para[22] = p_r_star;
    para[23] = sig_u 
    para[24] = sig_r
    para[25] = sig_s
    para[26] = sig_e
    para[27] = sig_a
    para[28] = sig_r_star
    para[29] = g

    return para


# In[4]:


def Solve_ZLB(para):
    # Define Variables
    psi = para[0];epsilon = para[1];alpha = para[2];sigma = para[3];rho = para[4];rho_k = para[5];delta = para[6];n1 = para[7];n2 = para[8];phi_b = para[9];phi_s= para[10];theta = para[11];phi_x = para[12];phi_pi = para[13];phi_t =para[14];rh0= para[15];h_w = para[16];h_r = 1-h_w;p_u = para[17];p_r = para[18];p_s = para[19];p_e = para[20];p_a = para[21];p_r_star = para[22];
    p_r_star = para[22];sig_u =para[23];sig_r = para[24];sig_s = para[25];sig_e = para[26];sig_a = para[27];sig_r_star = para[28]
    h_r = 1-h_w;
    r = 0;a = 1;s = 2;u = 3;e = 4;
    r_sh = 0;a_sh = 1;s_sh = 2;u_sh = 3;e_sh = 4;i_sh = 5
    b = 5;k = 6;q_km = 7;b_km = 8;w = 9;tau = 10;b_k = 11;i = 12
    y = 13;pi = 14;q = 15;q_k = 16
    C_bar = .6;K_bar = .4;Y_bar = 1;c1 = Y_bar/(Y_bar-K_bar);c2 = K_bar/(Y_bar-K_bar)
    # Set up nuisance parameters
    alpha_tilde = (1-(1-.99*theta)*(1-theta))**-1;gam = alpha*(psi**-1)*(-epsilon);v_1 = alpha_tilde*.99*theta;v_2 = alpha_tilde*((1-.99*theta)/(1-gam))
    phi_1 = (1-alpha*(psi**-1));phi_2 = alpha*(psi**-1)*delta*c1;phi_3 = alpha+(psi**-1)*alpha*delta*c2
    phi_4 = alpha*(psi**-1)*h_w;Q = (C_bar**delta)/((sigma*K_bar)**(n2))
    # Generate Matrices
    GAM0 = np.zeros([17,17]);GAM1 = np.zeros([17,17]);PSI = np.zeros([17,6])    
    # EQ 0 Goverment Bond Demand
    GAM0[b,b] =1
    GAM1[b,b] =1
    GAM0[b,q] = (1-rho)
    GAM0[b,pi] =1
    GAM0[b,tau] = 1
    GAM0[b,s] = 1
    # EQ 1 Capital Evolution 
    GAM0[k,k] =1
    GAM1[k,k] =1-sigma
    GAM0[k,b_km] =-1*( 1-2*sigma)
    GAM0[k,q_km] =-1*(1-2*sigma)
    GAM1[k,b_km] = -2*sigma
    GAM1[k,q_km] = -2*sigma
    GAM1[k,e] = -2*sigma
    # EQ 2 tech shock
    GAM0[a,a] = 1
    GAM1[a,a] = p_a
    PSI[a,a_sh] = sig_a
    # EQ 3 Seinorage shock
    GAM0[s,s] = 1 
    GAM1[s,s] = p_s
    PSI[s,s_sh] = sig_s
    # EQ 4 Demand shock
    GAM0[r,r] = 1
    GAM0[r,i] = p_r_star
    GAM1[r,r] = p_r 
    PSI[r,r_sh] = sig_r 
    # EQ 5 Cost Push shock
    GAM0[u,u] = 1 
    GAM1[u,u] = p_u 
    PSI[u,u_sh] = sig_u 
    # EQ 7 Investment "Credit Freeze" shock
    GAM0[e,e] = 1 
    GAM1[e,e] = p_e 
    PSI[e,e_sh] = sig_e 
    # EQ 8 coporate bond price lag
    GAM0[q_km,q_km] = 1
    GAM1[q_km,q_k] = 1
    # EQ 9 coporate bond demand lag
    GAM0[b_km,b_km] = 1
    GAM1[b_km,b_k] = 1
    # EQ 10 Money/IS equation
    GAM1[y,y] = c1
    GAM1[y,k] = c2
    GAM0[y,y] = c1*delta**-1
    GAM0[y,k] = -c2*delta**-1
    GAM0[y,pi] = delta**-1
    GAM0[y,tau] = -h_r*delta**-1
    #GAM1[y,i] = delta**-1
    GAM1[y,r] = -delta**-1
    # EQ 11 Phillips Curve
    GAM1[pi,pi] = 1
    GAM0[pi,pi] =v_1
    GAM1[pi,w] = -v_2*phi_1
    GAM1[pi,y] = -v_2*phi_2
    GAM1[pi,k] = v_2*phi_3
    GAM0[pi,tau] = v_2*phi_4
    GAM0[pi,u] =1
    # EQ 12 Gov Bond Pricing
    GAM1[q,q] = n1
    GAM1[q,b] = n1
    GAM1[q,y] = -delta*c1
    GAM1[q,k] = delta*c2
    GAM0[q,y] = .99*delta*c1
    GAM0[q,k] = -.99*delta*c2
    GAM0[q,pi] = .99*delta
    GAM0[q,tau] = h_r
    GAM0[q,q] = -.99*rho
    # EQ 13 Corp Bond Pricing
    GAM1[q_k,q] = 1-n2-Q**-1
    GAM1[q_k,b] = n2
    GAM1[q_k,y] = -delta*c1*(Q**-1)
    GAM1[q_k,k] = delta*c2*(Q**-1)
    GAM0[q_k,y] = .99*delta*c1*.994*(Q**-1)
    GAM0[q_k,k] = -.99*delta*c2*.994*(Q**-1)
    GAM0[q_k,pi] = .99*delta*.994*(Q**-1)
    GAM0[q_k,tau] = h_r*.994*(Q**-1)
    GAM0[q_k,q_k] = -(.99*rho_k*.994-.99*rho_k*.42*.006)*(Q**-1)
    # EQ 14 Monetary Policy
    GAM0[i,i] = 1
    GAM1[i,i] = 1
    PSI[i,i_sh] =sig_r_star
    # EQ 15 Wages
    GAM0[w,w] = 1-alpha*psi
    GAM0[w,q_k] = -2*alpha*psi
    GAM0[w,b_k] = -2*alpha*psi
    GAM1[w,q_k] = -2*alpha*psi
    GAM1[w,b_k] = -2*alpha*psi
    GAM0[w,e] = -2*alpha*psi
    GAM0[w,y] = -(delta-psi)*c1
    GAM0[w,a] = -psi
    GAM0[w,k] = delta*c2 
    GAM0[w,tau] = h_w 
    # EQ 16 Tax LOM
    GAM0[tau,tau] =1    
    GAM0[tau,b] = -phi_b
    GAM0[tau,q] = -phi_b 
    GAM0[tau,s] = phi_s 
    GAM1[tau,tau] = phi_t 

    # EQ 17 Corporate Bond Demand
    GAM0[b_k,b_k] = 1
    GAM0[b_k,w] = -1
    GAM1[b_k,q_k] = 1
    GAM1[b_k,b_k] = 1
    GAM0[b_k,y] = -(1-alpha)**-1
    GAM0[b_k,a] = (1-alpha)**-1
    GAM0[b_k,k] = (1-alpha)**-1
    GAM0[b_k,e] =  1
    GAM0[b_k,q_k] = 1

    n_s = 13;n_j = 4
    VV = np.zeros(17)
    F = m(inv(GAM0),GAM1);e_vals, e_vecs = eig(F)
    for i in range(17):
        VV[i] =  ((np.real(e_vals[i])**2)+(np.imag(e_vals[i])**2))**.5
    idx = np.argsort(VV)
    H = e_vecs;H = inv(e_vecs[:,idx]);G_tilde = m(H,PSI)
    # Partion Matrices (Imaginary Component)
    H11 = H[0:n_s,0:n_s];H12 = H[0:n_s,n_s:];H21 = H[n_s:,0:n_s];H22 = H[n_s:,n_s:]
    F11 = np.real(F[0:n_s,0:n_s]);F22 = np.real(F[0:n_s,n_s:]);
    L1 = np.diagflat(e_vals[idx][0:n_s]);L2 = np.diagflat(e_vals[idx][n_s:])
    G1 = PSI[0:n_s,:];G2 = PSI[n_s:,:];G1_tilde= G_tilde[0:n_s,:];G2_tilde= G_tilde[n_s:,:]
    try:
        Q = inv(H22)
    except:
        H22+=1e-2
        Q = inv(H22)

    # Coef for Jumps
    B = np.real(m(m(-inv(Q),inv(L2)),G2_tilde));A = np.real(m(-inv(Q),H21))
    # Coef for States
    C = np.real(F11 -m(F22,m(Q,H21)));D = np.real(m(F22,m(m(-Q,inv(L2)),G2_tilde))+G1)
    # Y_t = A*X_t + B*eps_t 
    # X_t = C*X_t-1 + D*eps_t    
    print(np.count_nonzero(abs(VV)>=1));print('sien on output: ', A[0,s]>0);print('sien on inflation: ', A[1,s]>0);print('sien on gov bond price: ', A[2,s]>0);print('sien on private bond price: ', A[3,s]>0)
    
    A1 = A[:,5:];A2 = A[:,0:5];C1 = C[5:,5:];C2 = C[5:,0:5]
    A1[:,7] = 0;C1[5:,7] = 0 

    R = np.zeros([5,5])
    R[0,0] =p_r
    R[1,1] =p_a
    R[2,2] =p_s
    R[3,3] =p_u
    R[4,4] =p_e

    # Learning Block
    A11 = A[:,0:5]
    A12 = A[:,5:]
    C11 = C[5:,0:5]
    C12 = C[5:,5:]
    

    
    return R,A1,A2,C1,C2,PSI[:,:-1],A11,A12,C11,C12


# In[5]:


def Solve(para):
    # Define Variables
    psi = para[0];epsilon = para[1];alpha = para[2];sigma = para[3];rho = para[4];rho_k = para[5];delta = para[6];n1 = para[7];n2 = para[8];phi_b = para[9];phi_s= para[10];theta = para[11];phi_x = para[12];phi_pi = para[13];phi_t =para[14];rh0= para[15];h_w = para[16];h_r = 1-h_w;p_u = para[17];p_r = para[18];p_s = para[19];p_e = para[20];p_a = para[21];
    p_r_star = para[22];sig_u =para[23];sig_r = para[24];sig_s = para[25];sig_e = para[26];sig_a = para[27];sig_r_star = para[28]
    r = 0;a = 1;s = 2;u = 3;e = 4;
    r_sh = 0;a_sh = 1;s_sh = 2;u_sh = 3;e_sh = 4;
    b = 5;k = 6;q_km = 7;b_km = 8;w = 9;tau = 10;b_k = 11;i = 12
    y = 13;pi = 14;q = 15;q_k = 16
    C_bar = .6;K_bar = .4;Y_bar = 1;c1 = Y_bar/(Y_bar-K_bar);c2 = K_bar/(Y_bar-K_bar)
    # Set up nuisance parameters
    alpha_tilde = (1-(1-.99*theta)*(1-theta))**-1;gam = alpha*(psi**-1)*(-epsilon);v_1 = alpha_tilde*.99*theta;v_2 = alpha_tilde*((1-.99*theta)/(1-gam))
    phi_1 = (1-alpha*(psi**-1));phi_2 = alpha*(psi**-1)*delta*c1;phi_3 = alpha+(psi**-1)*alpha*delta*c2
    phi_4 = alpha*(psi**-1)*h_w;Q = (C_bar**delta)/((sigma*K_bar)**(n2))
    # Generate Matrices
    GAM0 = np.zeros([17,17]);GAM1 = np.zeros([17,17]);PSI = np.zeros([17,5])    
    # EQ 0 Goverment Bond Demand
    GAM0[b,b] =1
    GAM1[b,b] =1
    GAM0[b,q] = (1-rho)
    GAM0[b,pi] =1
    GAM0[b,tau] = 1
    GAM0[b,s] = 1
    # EQ 1 Capital Evolution 
    GAM0[k,k] =1
    GAM1[k,k] =1-sigma
    GAM0[k,b_km] =-1*( 1-2*sigma)
    GAM0[k,q_km] =-1*(1-2*sigma)
    GAM1[k,b_km] = -2*sigma
    GAM1[k,q_km] = -2*sigma
    GAM1[k,e] = -2*sigma
    # EQ 2 tech shock
    GAM0[a,a] = 1
    GAM1[a,a] = p_a
    PSI[a,a_sh] = sig_a
    # EQ 3 Seinorage shock
    GAM0[s,s] = 1 
    GAM1[s,s] = p_s
    PSI[s,s_sh] = sig_s
    # EQ 4 Demand shock
    GAM0[r,r] = 1 
    GAM1[r,r] = p_r 
    PSI[r,r_sh] = sig_r 
    # EQ 5 Cost Push shock
    GAM0[u,u] = 1 
    GAM1[u,u] = p_u 
    PSI[u,u_sh] = sig_u 
    # EQ 7 Investment "Credit Freeze" shock
    GAM0[e,e] = 1 
    GAM1[e,e] = p_e
    PSI[e,e_sh] = sig_e
    # EQ 8 coporate bond price lag
    GAM0[q_km,q_km] = 1
    GAM1[q_km,q_k] = 1
    # EQ 9 coporate bond demand lag
    GAM0[b_km,b_km] = 1
    GAM1[b_km,b_k] = 1
    # EQ 10 Money/IS equation
    GAM1[y,y] = c1
    GAM1[y,k] = c2
    GAM0[y,y] = c1*delta**-1
    GAM0[y,k] = -c2*delta**-1
    GAM0[y,pi] = delta**-1
    GAM0[y,tau] = -h_r*delta**-1
    GAM1[y,i] = delta**-1
    GAM1[y,r] = -delta**-1
    # EQ 11 Phillips Curve
    GAM1[pi,pi] = 1
    GAM0[pi,pi] =v_1
    GAM1[pi,w] = -v_2*phi_1
    GAM1[pi,y] = -v_2*phi_2
    GAM1[pi,k] = v_2*phi_3
    GAM0[pi,tau] = v_2*phi_4
    GAM0[pi,u] =1
    # EQ 12 Gov Bond Pricing
    GAM1[q,q] = n1
    GAM1[q,b] = n1
    GAM1[q,y] = -delta*c1
    GAM1[q,k] = delta*c2
    GAM0[q,y] = .99*delta*c1
    GAM0[q,k] = -.99*delta*c2
    GAM0[q,pi] = .99*delta
    GAM0[q,tau] = h_r
    GAM0[q,q] = -.99*rho
    # EQ 13 Corp Bond Pricing
    GAM1[q_k,q] = 1-n2-Q**-1
    GAM1[q_k,b] = n2
    GAM1[q_k,y] = -delta*c1*(Q**-1)
    GAM1[q_k,k] = delta*c2*(Q**-1)
    GAM0[q_k,y] = .99*delta*c1*.994*(Q**-1)
    GAM0[q_k,k] = -.99*delta*c2*.994*(Q**-1)
    GAM0[q_k,pi] = .99*delta*.994*(Q**-1)
    GAM0[q_k,tau] = h_r*.994*(Q**-1)
    GAM0[q_k,q_k] = -(.99*rho_k*.994-.99*rho_k*.42*.006)*(Q**-1)
    # EQ 14 Monetary Policy
    GAM0[i,i] =1
    GAM0[i,y] =-phi_x
    GAM0[i,pi] =-phi_pi
    GAM1[i,i] =rh0
    # EQ 15 Wages
    GAM0[w,w] = 1-alpha*psi
    GAM0[w,q_k] = -2*alpha*psi
    GAM0[w,b_k] = -2*alpha*psi
    GAM1[w,q_k] = -2*alpha*psi
    GAM1[w,b_k] = -2*alpha*psi
    GAM0[w,e] = -2*alpha*psi
    GAM0[w,y] = -(delta-psi)*c1
    GAM0[w,a] = -psi
    GAM0[w,k] = delta*c2 
    GAM0[w,tau] = h_w 
    # EQ 16 Tax LOM
    GAM0[tau,tau] =1    
    GAM0[tau,b] = -phi_b
    GAM0[tau,q] = -phi_b 
    GAM0[tau,s] = phi_s 
    GAM1[tau,tau] = phi_t 

    # EQ 17 Corporate Bond Demand
    GAM0[b_k,b_k] = 1
    GAM0[b_k,w] = -1
    GAM1[b_k,q_k] = 1
    GAM1[b_k,b_k] = 1
    GAM0[b_k,y] = -(1-alpha)**-1
    GAM0[b_k,a] = (1-alpha)**-1
    GAM0[b_k,k] = (1-alpha)**-1
    GAM0[b_k,e] =  1
    GAM0[b_k,q_k] = 1

    n_s = 13
    n_j = 4
    VV = np.zeros(17)
    F = m(inv(GAM0),GAM1)
    e_vals, e_vecs = eig(F)
    for i in range(17):
        VV[i] =  ((np.real(e_vals[i])**2)+(np.imag(e_vals[i])**2))**.5
    idx = np.argsort(VV)
    e_vals, e_vecs = eig(F)

    H = e_vecs
    H = inv(e_vecs[:,idx])
    G_tilde = m(H,PSI)
    # Partion Matrices (Imaginary Component)
    H11 = H[0:n_s,0:n_s]
    H12 = H[0:n_s,n_s:]
    H21 = H[n_s:,0:n_s]
    H22 = H[n_s:,n_s:]
    F11 = np.real(F[0:n_s,0:n_s])
    F22 = np.real(F[0:n_s,n_s:])
    L1 = np.diagflat(e_vals[idx][0:n_s])
    L2 = np.diagflat(e_vals[idx][n_s:])
    G1 = PSI[0:n_s,:]
    G2 = PSI[n_s:,:]
    G1_tilde= G_tilde[0:n_s,:]
    G2_tilde= G_tilde[n_s:,:]
    #Q = inv(H22+1e-30*np.random.standard_normal([n_j,n_j]))
    Q = inv(H22)
    # Coef for Jumps
    B = np.real(m(m(-inv(Q),inv(L2)),G2_tilde))
    A = -np.real(m(inv(Q),H21))
    # Coef for States
    C = np.real(F11 -m(F22,m(Q,H21)))
    D = np.real(m(F22,m(m(-Q,inv(L2)),G2_tilde))+G1)
    # Y_t = A*X_t + B*eps_t 
    # X_t = C*X_t-1 + D*eps_t    
    print(np.count_nonzero(abs(VV)>=1))
    print('sien on output: ', A[0,s]>0)
    print('sien on inflation: ', A[1,s]>0)
    print('sien on gov bond price: ', A[2,s]>0)
    print('sien on private bond price: ', A[3,s]>0)
    
    A1 = A[:,5:]
    A2 = A[:,0:5]
    C1 = C[5:,5:]
    C2 = C[5:,0:5]
    
    

    R = np.zeros([5,5])
    R[0,0] =p_r
    R[1,1] =p_a
    R[2,2] =p_s
    R[3,3] =p_u
    R[4,4] =p_e

    A11 = A[:,0:5]
    A12 = A[:,5:]
    C11 = C[5:,0:5]
    C12 = C[5:,5:]
    
    
    
    return R,A1,A2,C1,C2,PSI,A11,A12,C11,C12


# In[6]:


def Set_M(para,PSI,aa):
    # Set Variables
    y = 0;pi = 1;q = 2;q_k = 3;
    b = 0;k = 1;q_km = 2;b_km = 3;w = 4;tau = 5;b_k = 6;i = 7;
    r = 0;a = 1;s = 2;u = 3;e = 4;
    p_r_star = para[22];sig_u =para[23];sig_r = para[24];sig_s = para[25];sig_e = para[26];sig_a = para[27];sig_r_star = para[28]


    # Initialize Parameters
    psi = para[0];epsilon = para[1];alpha = para[2];sigma = para[3];rho = para[4];rho_k = para[5];delta = para[6];n1 = para[7];n2 = para[8];phi_b = para[9];phi_s= para[10];theta = para[11];phi_x = para[12];phi_pi = para[13];phi_t =para[14];rh0= para[15];h_w = para[16];h_r = 1-h_w;p_u = para[17];p_r = para[18];p_s = para[19];p_e = para[20];p_a = para[21];
    # Set up nuisance parameters
    C_bar = .4;K_bar = .24;Y_bar = .4+.24;c1 = Y_bar/(Y_bar-K_bar);c2 = K_bar/(Y_bar-K_bar)
    alpha_tilde = (1-(1-.99*theta)*(1-theta))**-1;gam = alpha*(psi**-1)*(-epsilon);v_1 = alpha_tilde*.99*theta;v_2 = alpha_tilde*((1-.99*theta)/(1-gam))
    phi_1 = (1-alpha*(psi**-1));phi_2 = alpha*(psi**-1)*delta*c1;phi_3 = alpha+(psi**-1)*alpha*delta*c2
    phi_4 = alpha*(psi**-1)*h_w;Q = (C_bar**delta)/((sigma*K_bar)**(n2))
    z1 = (c1**-1)*c2-(c1**-1)*c2*(1-sigma);z2 = 1;z3 = (c1**-1)*(delta**-1)*(1+((1-phi_b)**-1)*phi_b*h_r);z4 = -(c1**-1)*(delta**-1)
    z5 = (c1**-1)*(delta**-1)*((1-phi_b)**-1)*(-phi_b+rho-1)*h_r;z6 = (c1**-1)*(delta**-1)*((1-phi_b)**-1)*(phi_b+phi_s)*p_s*h_r
    z7 = (c1**-1)*(delta**-1)*((1-phi_b)**-1)*(phi_t)*h_r;z8 = -(c1**-1)*(c2)*(1-2*sigma)*rho_k*(.996+.004*.42);
    z9 = -(c1**-1)*(c2)*(1-2*sigma);z10 = (c1**-1)*(c2)*(2*sigma);z11 = -delta**-1
    # Generate Matrices
    B10 = np.zeros([4,4]);B11 = np.zeros([4,4]);B12 = np.zeros([4,8]);B13 = np.zeros([4,5]);B14 = np.zeros([4,8]);B20 = np.zeros([8,8]);B21 = np.zeros([8,4]);B22 = np.zeros([8,8]);B23 = np.zeros([8,5])
    #jumps
    # Y
    B10[y,y] += c1;B12[y,k] += c2;B11[y,y] += c1
    B14[y,k] += -c2;B11[y,pi] += delta**-1;B14[y,tau] += -h_r*(delta**-1)
    B12[y,i] += -delta**-1;B13[y,r] += delta**-1
    # Pi
    B10[pi,pi] +=1
    B11[pi,pi] += v_1
    B12[pi,w] += v_2*phi_1
    B10[pi,y] += -v_2*phi_2
    B12[pi,k] += -v_2*phi_3
    B14[pi,tau] += v_2*phi_4
    # q
    B10[q,q] += n2
    B10[q,b] += n2
    B10[q,y] += -delta*c1
    B12[q,k] += -delta*c2
    B11[q,y] += .99*delta*c1
    B14[q,k] += -.99*delta*c2
    B11[q,pi] += .99*delta
    B14[q,tau] += h_r
    B11[q,q] += -rho*.99
    # q_k

    B10[q_k,q_k] += (1-n2-(Q**-1))
    
    B12[q_k,b_k] += n2
    B10[q_k,y] += delta*c1*(Q**-1)
    B12[q_k,k] +=delta*c2*(Q**-1)
    
    B11[q_k,y] += .99*delta*c1*(.994+.42*.006)*(Q**-1)
    B14[q_k,k] += -.99*delta*c2*(.994+.42*.006)*(Q**-1)
    B11[q_k,pi] += .99*delta*(.994+.42*.006)*(Q**-1)
    B14[q_k,tau] += .99*h_r*(.994+.42*.006)*(Q**-1)
    B11[q_k,q_k] += -(.99*rho_k)*(.994+.006*.42)*(Q**-1)
    # states
    #k

    B20[k,k] +=1
    B22[k,k] += 1-sigma
    B21[k,q_k] += (1-2*sigma)*rho_k*.996+.004*.42*(1-2*sigma)*rho_k  
    B20[k,b_k] += -(1-2*sigma) 
    B20[k,b_km] += -2*sigma
    B22[k,q_km] += -2*sigma
    B23[k,e] += -2*sigma
    #b

    B20[b,b] +=1
    B22[b,b] +=1
    B21[b,q] += (rho-1)*rho

    #B21[b,pi] =-1
    B21[b,pi] += -1*v_1
    B20[b,w] += v_2*phi_1
    #B20[b,y] = v_2*phi_2
    B20[b,k] += -v_2*phi_3
    #B23[b,tau] = -v_2*phi_4
    B21[b,q] += ((1-phi_b)**-1)*(phi_b+rho-1)*(-v_2*phi_4-1)
    B21[b,pi] += ((1-phi_b)**-1)*(-phi_b)*(-v_2*phi_4-1)
    B22[b,s] += ((1-phi_b)**-1)*(-phi_s-phi_b)*p_s*(-v_2*phi_4-1)
    B20[b,tau] += -((1-phi_b)**-1)*(-phi_t)*(-v_2*phi_4-1)
    # this went to above
    #B20[b,tau] += 1
    B20[b,s] += 1

    # q_km
    B20[q_km,q_km] += 1
    B21[q_km,q_k] += rho_k*rho_k
    # b_km
    B20[b_km,b_km] += 1
    B22[b_km,b_k] += 1
    #w

    B20[w,w] += 1-alpha*psi
    B20[w,k] += c2*delta

    #B20[w,tau] = h_w
    B21[w,q] += ((1-phi_b)**-1)*(phi_b+rho-1)*(-h_w)
    B21[w,pi] += ((1-phi_b)**-1)*(-phi_b)*(-h_w)
    B22[w,s] += ((1-phi_b)**-1)*(-phi_s-phi_b)*p_s*(-h_w)
    B20[w,tau] += ((1-phi_b)**-1)*(-phi_t)*(-h_w)

    B21[w,q_k] += 2*alpha*psi*(rho_k*.996+.004*.42*rho_k)
    B20[w,b_k] += -2*alpha*psi
    B22[w,b_k] += -2*alpha*psi
    B20[w,q_km] += 2*alpha*psi
    B23[w,e] += 2*alpha*psi
    B23[w,a] += psi
    #B20[w,y] = -(delta*c1-psi)


    #tau

    B20[tau,tau] +=1    
    B20[tau,b] += -phi_b
    B21[tau,q] += phi_b*rho
    B22[tau,tau] +=phi_t    
    B20[tau,s] +=phi_s 
    #b_k

    B20[b_k,b_k] += 1
    B20[b_k,w] += -1
    B20[b_k,q_km] += -1
    B22[b_k,b_k] += 1
    #B20[b_k,y] = -(1-alpha)**-1

    B23[b_k,a] += -(1-alpha)**-1
    B20[b_k,k] += (1-alpha)**-1
    B23[b_k,e] +=  -1
    B21[b_k,q_k] += -1*(rho_k*.996+rho_k*.004*.42)
    # i 

    B20[i,i] += 1
    #B20[i,pi] = -phi_pi
    B21[i,pi] += v_1*phi_pi
    B20[i,w] += -v_2*phi_1*phi_pi
    #B20[i,y] = -v_2*phi_2*phi_pi
    B20[i,k] += v_2*phi_3*phi_pi
    #B20[i,tau] = -v_2*phi_4*phi_pi
    B21[i,q] += ((1-phi_b)**-1)*(phi_b+rho-1)*(v_2*phi_4*phi_pi)
    B21[i,pi] += ((1-phi_b)**-1)*(-phi_b)*(v_2*phi_4*phi_pi)
    B22[i,s] += ((1-phi_b)**-1)*(-phi_s-phi_b)*p_s*(v_2*phi_4*phi_pi)
    B20[i,tau] += -((1-phi_b)**-1)*(-phi_t)*(v_2*phi_4*phi_pi)
    #B20[i,y] = -phi_x
    B22[i,i] +=rh0

    # Expression for Y in terms of States
    #B20[b_k,y] = -(1-alpha)**-1
    B20[b_k,k] += -z1*(1-alpha)**-1
    B21[b_k,y] += z2*(1-alpha)**-1
    B21[b_k,pi] += z3*(1-alpha)**-1
    B20[b_k,i] += -z4*(1-alpha)**-1
    B21[b_k,q] += z5*(1-alpha)**-1
    B20[b_k,s] += -z6*(1-alpha)**-1
    B20[b_k,tau] +=-z7*(1-alpha)**-1
    B21[b_k,q_k] += z8*(1-alpha)**-1
    B20[b_k,b_k] += -z9*(1-alpha)**-1
    B23[b_k,e] += z10*(1-alpha)**-1
    B23[b_k,r] += z11*(1-alpha)**-1
    #B20[w,y] = -(delta*c1-psi)
    B20[b_k,k] += -z1*(delta*c1-psi)
    B21[b_k,y] += z2*(delta*c1-psi)
    B21[b_k,pi] += z3*(delta*c1-psi)
    B20[b_k,i] += -z4*(delta*c1-psi)
    B21[b_k,q] += z5*(delta*c1-psi)
    B20[b_k,s] += -z6*(delta*c1-psi)
    B20[b_k,tau] +=-z7*(delta*c1-psi)
    B21[b_k,q_k] += z8*(delta*c1-psi)
    B20[b_k,b_k] += -z9*(delta*c1-psi)
    B23[b_k,e] += z10*(delta*c1-psi)
    B23[b_k,r] += z11*(delta*c1-psi)
    #B20[i,y] =-1*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,k] += -z1*(phi_x+v_2*phi_4*phi_pi)
    B21[b_k,y] += z2*(phi_x+v_2*phi_4*phi_pi)
    B21[b_k,pi] += z3*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,i] += -z4*(phi_x+v_2*phi_4*phi_pi)
    B21[b_k,q] += z5*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,s] += -z6*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,tau] +=-z7*(phi_x+v_2*phi_4*phi_pi)
    B21[b_k,q_k] += z8*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,b_k] += -z9*(phi_x+v_2*phi_4*phi_pi)
    B23[b_k,e] += z10*(phi_x+v_2*phi_4*phi_pi)
    B23[b_k,r] += z11*(phi_x+v_2*phi_4*phi_pi)
    #B20[b,y] = v_2*phi_2 
    B20[b_k,k] += -z1*(-v_2*phi_2)
    B21[b_k,y] += z2*(-v_2*phi_2)
    B21[b_k,pi] += z3*(-v_2*phi_2)
    B20[b_k,i] += -z4*(-v_2*phi_2)
    B21[b_k,q] += z5*(-v_2*phi_2)
    B20[b_k,s] += -z6*(-v_2*phi_2)
    B20[b_k,tau] +=-z7*(-v_2*phi_2)
    B21[b_k,q_k] += z8*(-v_2*phi_2)
    B20[b_k,b_k] += -z9*(-v_2*phi_2)
    B23[b_k,e] += z10*(-v_2*phi_2)
    B23[b_k,r] += z11*(-v_2*phi_2)

    B11 = m(inv(B10),B11)*aa;B12 = m(inv(B10),B12)*aa;B13 = m(inv(B10),B13)*aa;B14 = m(inv(B10),B14)*aa;B21 = m(inv(B20),B21)*aa;B22 = m(inv(B20),B22)*aa;B23 = m(inv(B20),B23)*aa
    PSI_tilde=PSI[0:5]*aa
    return B11,B12,B13,B14,B21,B22,B23,PSI_tilde


# In[7]:


def Set_M(para,PSI,aa):
    # Set Variables
    y = 0;pi = 1;q = 2;q_k = 3;
    b = 0;k = 1;q_km = 2;b_km = 3;w = 4;tau = 5;b_k = 6;i = 7;
    r = 0;a = 1;s = 2;u = 3;e = 4;


    # Initialize Parameters
    psi = para[0];epsilon = para[1];alpha = para[2];sigma = para[3];rho = para[4];rho_k = para[5];delta = para[6];n1 = para[7];n2 = para[8];phi_b = para[9];phi_s= para[10];theta = para[11];phi_x = para[12];phi_pi = para[13];phi_t =para[14];rh0= para[15];h_w = para[16];h_r = 1-h_w;p_u = para[17];p_r = para[18];p_s = para[19];p_e = para[20];p_a = para[21];
    # Set up nuisance parameters
    C_bar = .4;K_bar = .24;Y_bar = .4+.24;c1 = Y_bar/(Y_bar-K_bar);c2 = K_bar/(Y_bar-K_bar)
    alpha_tilde = (1-(1-.99*theta)*(1-theta))**-1;gam = alpha*(psi**-1)*(-epsilon);v_1 = alpha_tilde*.99*theta;v_2 = alpha_tilde*((1-.99*theta)/(1-gam))
    phi_1 = (1-alpha*(psi**-1));phi_2 = alpha*(psi**-1)*delta*c1;phi_3 = alpha+(psi**-1)*alpha*delta*c2
    phi_4 = alpha*(psi**-1)*h_w;Q = (C_bar**delta)/((sigma*K_bar)**(n2))
    z1 = (c1**-1)*c2-(c1**-1)*c2*(1-sigma);z2 = 1;z3 = (c1**-1)*(delta**-1)*(1+((1-phi_b)**-1)*phi_b*h_r);z4 = -(c1**-1)*(delta**-1)
    z5 = (c1**-1)*(delta**-1)*((1-phi_b)**-1)*(-phi_b+rho-1)*h_r;z6 = (c1**-1)*(delta**-1)*((1-phi_b)**-1)*(phi_b+phi_s)*p_s*h_r
    z7 = (c1**-1)*(delta**-1)*((1-phi_b)**-1)*(phi_t)*h_r;z8 = -(c1**-1)*(c2)*(1-2*sigma)*rho_k*(.996+.004*.42);
    z9 = -(c1**-1)*(c2)*(1-2*sigma);z10 = (c1**-1)*(c2)*(2*sigma);z11 = -delta**-1
    # Generate Matrices
    B10 = np.zeros([4,4]);B11 = np.zeros([4,4]);B12 = np.zeros([4,8]);B13 = np.zeros([4,5]);B14 = np.zeros([4,8]);B20 = np.zeros([8,8]);B21 = np.zeros([8,4]);B22 = np.zeros([8,8]);B23 = np.zeros([8,5])
    #jumps
    # Y
    B10[y,y] += c1;B12[y,k] += c2;B11[y,y] += c1
    B14[y,k] += -c2;B11[y,pi] += delta**-1;B14[y,tau] += -h_r*(delta**-1)
    B12[y,i] += -delta**-1;B13[y,r] += delta**-1
    # Pi
    B10[pi,pi] +=1
    B11[pi,pi] += v_1
    B12[pi,w] += v_2*phi_1
    B10[pi,y] += -v_2*phi_2
    B12[pi,k] += -v_2*phi_3
    B14[pi,tau] += v_2*phi_4
    # q
    B10[q,q] += n2
    B10[q,b] += n2
    B10[q,y] += -delta*c1
    B12[q,k] += -delta*c2
    B11[q,y] += .99*delta*c1
    B14[q,k] += -.99*delta*c2
    B11[q,pi] += .99*delta
    B14[q,tau] += h_r
    B11[q,q] += -rho*.99
    # q_k

    B10[q_k,q_k] += (1-n2-(Q**-1))
    
    B12[q_k,b_k] += n2
    B10[q_k,y] += delta*c1*(Q**-1)
    B12[q_k,k] +=delta*c2*(Q**-1)
    
    B11[q_k,y] += .99*delta*c1*(.994+.42*.006)*(Q**-1)
    B14[q_k,k] += -.99*delta*c2*(.994+.42*.006)*(Q**-1)
    B11[q_k,pi] += .99*delta*(.994+.42*.006)*(Q**-1)
    B14[q_k,tau] += .99*h_r*(.994+.42*.006)*(Q**-1)
    B11[q_k,q_k] += -(.99*rho_k)*(.994+.006*.42)*(Q**-1)
    # states
    #k

    B20[k,k] +=1
    B22[k,k] += 1-sigma
    B21[k,q_k] += (1-2*sigma)*rho_k*.996+.004*.42*(1-2*sigma)*rho_k  
    B20[k,b_k] += -(1-2*sigma) 
    B20[k,b_km] += -2*sigma
    B22[k,q_km] += -2*sigma
    B23[k,e] += -2*sigma
    #b

    B20[b,b] +=1
    B22[b,b] +=1
    B21[b,q] += (rho-1)*rho

    #B21[b,pi] =-1
    B21[b,pi] += -1*v_1
    B20[b,w] += v_2*phi_1
    #B20[b,y] = v_2*phi_2
    B20[b,k] += -v_2*phi_3
    #B23[b,tau] = -v_2*phi_4
    B21[b,q] += ((1-phi_b)**-1)*(phi_b+rho-1)*(-v_2*phi_4-1)
    B21[b,pi] += ((1-phi_b)**-1)*(-phi_b)*(-v_2*phi_4-1)
    B22[b,s] += ((1-phi_b)**-1)*(-phi_s-phi_b)*p_s*(-v_2*phi_4-1)
    B20[b,tau] += -((1-phi_b)**-1)*(-phi_t)*(-v_2*phi_4-1)
    # this went to above
    #B20[b,tau] += 1
    B20[b,s] += 1

    # q_km
    B20[q_km,q_km] += 1
    B21[q_km,q_k] += rho_k*rho_k
    # b_km
    B20[b_km,b_km] += 1
    B22[b_km,b_k] += 1
    #w

    B20[w,w] += 1-alpha*psi
    B20[w,k] += c2*delta

    #B20[w,tau] = h_w
    B21[w,q] += ((1-phi_b)**-1)*(phi_b+rho-1)*(-h_w)
    B21[w,pi] += ((1-phi_b)**-1)*(-phi_b)*(-h_w)
    B22[w,s] += ((1-phi_b)**-1)*(-phi_s-phi_b)*p_s*(-h_w)
    B20[w,tau] += ((1-phi_b)**-1)*(-phi_t)*(-h_w)

    B21[w,q_k] += 2*alpha*psi*(rho_k*.996+.004*.42*rho_k)
    B20[w,b_k] += -2*alpha*psi
    B22[w,b_k] += -2*alpha*psi
    B20[w,q_km] += 2*alpha*psi
    B23[w,e] += 2*alpha*psi
    B23[w,a] += psi
    #B20[w,y] = -(delta*c1-psi)


    #tau

    B20[tau,tau] +=1    
    B20[tau,b] += -phi_b
    B21[tau,q] += phi_b*rho
    B22[tau,tau] +=phi_t    
    B20[tau,s] +=phi_s 
    #b_k

    B20[b_k,b_k] += 1
    B20[b_k,w] += -1
    B20[b_k,q_km] += -1
    B22[b_k,b_k] += 1
    #B20[b_k,y] = -(1-alpha)**-1

    B23[b_k,a] += -(1-alpha)**-1
    B20[b_k,k] += (1-alpha)**-1
    B23[b_k,e] +=  -1
    B21[b_k,q_k] += -1*(rho_k*.996+rho_k*.004*.42)
    # i 

    B20[i,i] += 1
    #B20[i,pi] = -phi_pi
    B21[i,pi] += v_1*phi_pi
    B20[i,w] += -v_2*phi_1*phi_pi
    #B20[i,y] = -v_2*phi_2*phi_pi
    B20[i,k] += v_2*phi_3*phi_pi
    #B20[i,tau] = -v_2*phi_4*phi_pi
    B21[i,q] += ((1-phi_b)**-1)*(phi_b+rho-1)*(v_2*phi_4*phi_pi)
    B21[i,pi] += ((1-phi_b)**-1)*(-phi_b)*(v_2*phi_4*phi_pi)
    B22[i,s] += ((1-phi_b)**-1)*(-phi_s-phi_b)*p_s*(v_2*phi_4*phi_pi)
    B20[i,tau] += -((1-phi_b)**-1)*(-phi_t)*(v_2*phi_4*phi_pi)
    #B20[i,y] = -phi_x
    B22[i,i] +=rh0

    # Expression for Y in terms of States
    #B20[b_k,y] = -(1-alpha)**-1
    B20[b_k,k] += -z1*(1-alpha)**-1
    B21[b_k,y] += z2*(1-alpha)**-1
    B21[b_k,pi] += z3*(1-alpha)**-1
    B20[b_k,i] += -z4*(1-alpha)**-1
    B21[b_k,q] += z5*(1-alpha)**-1
    B20[b_k,s] += -z6*(1-alpha)**-1
    B20[b_k,tau] +=-z7*(1-alpha)**-1
    B21[b_k,q_k] += z8*(1-alpha)**-1
    B20[b_k,b_k] += -z9*(1-alpha)**-1
    B23[b_k,e] += z10*(1-alpha)**-1
    B23[b_k,r] += z11*(1-alpha)**-1
    #B20[w,y] = -(delta*c1-psi)
    B20[b_k,k] += -z1*(delta*c1-psi)
    B21[b_k,y] += z2*(delta*c1-psi)
    B21[b_k,pi] += z3*(delta*c1-psi)
    B20[b_k,i] += -z4*(delta*c1-psi)
    B21[b_k,q] += z5*(delta*c1-psi)
    B20[b_k,s] += -z6*(delta*c1-psi)
    B20[b_k,tau] +=-z7*(delta*c1-psi)
    B21[b_k,q_k] += z8*(delta*c1-psi)
    B20[b_k,b_k] += -z9*(delta*c1-psi)
    B23[b_k,e] += z10*(delta*c1-psi)
    B23[b_k,r] += z11*(delta*c1-psi)
    #B20[i,y] =-1*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,k] += -z1*(phi_x+v_2*phi_4*phi_pi)
    B21[b_k,y] += z2*(phi_x+v_2*phi_4*phi_pi)
    B21[b_k,pi] += z3*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,i] += -z4*(phi_x+v_2*phi_4*phi_pi)
    B21[b_k,q] += z5*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,s] += -z6*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,tau] +=-z7*(phi_x+v_2*phi_4*phi_pi)
    B21[b_k,q_k] += z8*(phi_x+v_2*phi_4*phi_pi)
    B20[b_k,b_k] += -z9*(phi_x+v_2*phi_4*phi_pi)
    B23[b_k,e] += z10*(phi_x+v_2*phi_4*phi_pi)
    B23[b_k,r] += z11*(phi_x+v_2*phi_4*phi_pi)
    #B20[b,y] = v_2*phi_2 
    B20[b_k,k] += -z1*(-v_2*phi_2)
    B21[b_k,y] += z2*(-v_2*phi_2)
    B21[b_k,pi] += z3*(-v_2*phi_2)
    B20[b_k,i] += -z4*(-v_2*phi_2)
    B21[b_k,q] += z5*(-v_2*phi_2)
    B20[b_k,s] += -z6*(-v_2*phi_2)
    B20[b_k,tau] +=-z7*(-v_2*phi_2)
    B21[b_k,q_k] += z8*(-v_2*phi_2)
    B20[b_k,b_k] += -z9*(-v_2*phi_2)
    B23[b_k,e] += z10*(-v_2*phi_2)
    B23[b_k,r] += z11*(-v_2*phi_2)

    B11 = m(inv(B10),B11)*aa;B12 = m(inv(B10),B12)*aa;B13 = m(inv(B10),B13)*aa;B14 = m(inv(B10),B14)*aa;B21 = m(inv(B20),B21)*aa;B22 = m(inv(B20),B22)*aa;B23 = m(inv(B20),B23)*aa
    PSI_tilde=PSI[0:5]*aa
    return B11,B12,B13,B14,B21,B22,B23,PSI_tilde
# Initialize Parameters
psi = 1.6**-1;epsilon = 1.2;alpha = .3;sigma = .015;rho = .9;rho_k = .93;
delta = 2.3**-1;n1 = 1.1**-1;n2 = 1.1**-1
phi_b = 1.5;phi_s= 0.1;theta = .016;
phi_x = 0.5;phi_pi = 1.5;phi_t =.2;
rh0= .7;h_w = .5;h_r = 1-h_w
p_u = .5;p_r = .5;p_s = .5;p_e = .5;p_a = .5;g = 0.03

#### Nuisance parameters ########
r_sh = 0;a_sh = 1;s_sh = 2;u_sh = 3;e_sh = 4;
C_bar = .60;K_bar = .4;Y_bar = 1;c1 = Y_bar/(Y_bar-K_bar);c2 = K_bar/(Y_bar-K_bar)
alpha_tilde = (1-(1-.99*theta)*(1-theta))**-1;gam = alpha*(psi**-1)*(-epsilon);v_1 = alpha_tilde*.99*theta;v_2 = alpha_tilde*((1-.99*theta)/(1-gam))
phi_1 = (1-alpha*(psi**-1));phi_2 = alpha*(psi**-1)*delta*c1;phi_3 = alpha+(psi**-1)*alpha*delta*c2
phi_4 = alpha*(psi**-1)*h_w;Q = (C_bar**delta)/((sigma*K_bar)**(n2))
z1 = (c1**-1)*c2-(c1**-1)*c2*(1-sigma);z2 = 1;z3 = (c1**-1)*(delta**-1)*(1+((1-phi_b)**-1)*phi_b*h_r);z4 = -(c1**-1)*(delta**-1)
z5 = (c1**-1)*(delta**-1)*((1-phi_b)**-1)*(-phi_b+rho-1)*h_r;z6 = (c1**-1)*(delta**-1)*((1-phi_b)**-1)*(phi_b+phi_s)*p_s*h_r
z7 = (c1**-1)*(delta**-1)*((1-phi_b)**-1)*(phi_t)*h_r;z8 = -(c1**-1)*(c2)*(1-2*sigma)*rho_k*(.996+.004*.42);
z9 = -(c1**-1)*(c2)*(1-2*sigma);z10 = (c1**-1)*(c2)*(2*sigma);z11 = -delta**-1


rho =.89 ;rho_k = .9059
x = .99*rho

print( 'duration gov bond: ', (x*(1-x)**-2)/12 )
xk = .99*rho_k
c1 = .997+.003*.42
print( 'duration corp bond: ', (xk*(1-xk)**-2)/12*c1 )
# scalar
aa = 1e-2
para = set_para()
R,A1_z,A2_z,C1_z,C2_z,PSI_z,A11_z,A12_z,C11_z,C12_z = Solve_ZLB(para);
R,A1,A2,C1,C2,PSI,A11,A12,C11,C12 = Solve(para);
B11,B12,B13,B14,B21,B22,B23,PSI_tilde = Set_M(para,PSI,aa);
PSI_tilde*=aa
C11  =cp.array(C11);C12 = cp.array(C12);
A11 = cp.array(A11);A12 = cp.array(A12);
A12_z = cp.array(A12_z);A11_z = cp.array(A11_z);
C11_z = cp.array(C11_z);C12_z = cp.array(C12_z);
PSI_tilde = cp.array(PSI_tilde);R = cp.array(R);
B11 = cp.array(B11);B12 = cp.array(B12);B13 = cp.array(B13);
B14 = cp.array(B14);B21 = cp.array(B21);B22 = cp.array(B22);B23 = cp.array(B23); 
m =cp.matmul
resh = cp.reshape
inv = cp.linalg.inv


# In[8]:


######################## Particle Filter Setup ########################################


# In[9]:


## Do the Proper Imports
import numpy as np;import pycuda
import pycuda.driver as drv;import pycuda.driver as cuda;import pycuda.autoinit
from pycuda.compiler import SourceModule;import os;import time;import pycuda.gpuarray as gpuarray
if os.system("cl.exe"):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64"
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")
os.environ['PATH']+= ';'+r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib\x64"
os.environ['PATH']+= ';'+r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin"


# In[10]:


# Initialize Dimensions
m = np.matmul
resh = np.reshape
inv = np.linalg.inv


# In[11]:


R,A1,A2,C1,C2,PSI,A11,A12,C11,C12 = Solve(para);


# In[ ]:





# In[12]:


# Create Host Memory 
def init_gpu(para,aa,J):
    n_phi = 156;n_s = 13;n_x = 4; n_para = 29;n_shocks = 5;T = 1;
    R,A1_z,A2_z,C1_z,C2_z,PSI_z,A11_z,A12_z,C11_z,C12_z = Solve_ZLB(para);
    R,A1,A2,C1,C2,PSI,A11,A12,C11,C12 = Solve(para);
    R = np.diagonal(R);PSI = np.diagonal(PSI);
    R = R.astype(np.float32);PSI = PSI.astype(np.float32)
    B11,B12,B13,B14,B21,B22,B23,PSI_tilde = Set_M(para,PSI,aa);
    B11 = B11.astype(np.float32);B12 = B12.astype(np.float32);B13 = B13.astype(np.float32);B14 = B14.astype(np.float32);
    B21 = B21.astype(np.float32);B22 =B22.astype(np.float32);B23 =B23.astype(np.float32)
    Q0 = C11*aa;Q1 = C12*aa

    G0 = A11*aa;G1 = A12*aa
    Q0_z = C11_z*aa;Q1_z = C12_z*aa
    G0_z = A11_z*aa;G1_z = A12_z*aa
    U = 0*np.random.standard_normal([n_shocks,J,1]);EPS = np.random.standard_normal([n_shocks,J,T]).astype(np.float32)
    Y = np.random.standard_normal([n_x,J]);M = np.random.standard_normal([n_s,J])
    U = U.astype(np.float32);EPS = EPS.astype(np.float32)
    Y = Y.astype(np.float32);M = M.astype(np.float32)
    EX = np.zeros([n_x,J]);EX1 = EX
    EM = np.zeros([n_s,J]);EM1 = EM
    EX = EX.astype(np.float32);EX1 = EX1.astype(np.float32)
    EM = EM.astype(np.float32);EM1 = EM1.astype(np.float32)

    phi_z = np.zeros([156,J])+np.vstack([resh(Q0,[-1,1]),resh(Q1,[-1,1]),resh(G1,[-1,1]),resh(G0,[-1,1])]);
    phi_n =np.zeros([156,J]) +np.vstack([resh(Q0_z,[-1,1]),resh(Q1_z,[-1,1]),resh(G1_z,[-1,1]),resh(G0_z,[-1,1])]);
    phi_z = resh(phi_z,[156,J,1]);phi_n = resh(phi_n,[156,J,1]);
    phi_n[0,:,:] = resh(np.arange(J),[3,1])
    phi_z = phi_z.astype(np.float32);phi_n = phi_n.astype(np.float32);RR = 1e-2*np.eye(n_phi).astype(np.float32)
    # Create Device Memory
    U_gpu = cuda.mem_alloc(U.nbytes);EPS_gpu = cuda.mem_alloc(EPS.nbytes)
    Y_gpu = cuda.mem_alloc(Y.nbytes);M_gpu = cuda.mem_alloc(M.nbytes)
    EX_gpu = cuda.mem_alloc(EX.nbytes);EX1_gpu = cuda.mem_alloc(EX1.nbytes)
    EM_gpu = cuda.mem_alloc(EM.nbytes);EM1_gpu = cuda.mem_alloc(EM1.nbytes)
    phi_z_gpu = cuda.mem_alloc(phi_z.nbytes);phi_n_gpu = cuda.mem_alloc(phi_n.nbytes)
    RR_gpu = cuda.mem_alloc(RR.nbytes);PSI_gpu =  cuda.mem_alloc(PSI.nbytes)
    R_gpu = cuda.mem_alloc(R.nbytes);
    B11_gpu = cuda.mem_alloc(B11.nbytes);B12_gpu = cuda.mem_alloc(B12.nbytes);
    B13_gpu = cuda.mem_alloc(B13.nbytes);B14_gpu = cuda.mem_alloc(B14.nbytes);
    B21_gpu = cuda.mem_alloc(B21.nbytes);B22_gpu = cuda.mem_alloc(B22.nbytes);B23_gpu = cuda.mem_alloc(B23.nbytes);
    Q0_gpu =  cuda.mem_alloc(Q0.nbytes);Q1_gpu =  cuda.mem_alloc(Q1.nbytes);
    G0_gpu =  cuda.mem_alloc(G0.nbytes);G1_gpu =  cuda.mem_alloc(G1.nbytes);
    # Copy Host Memory to GPU
    cuda.memcpy_htod(U_gpu,U);cuda.memcpy_htod(EPS_gpu, EPS)
    cuda.memcpy_htod(Y_gpu,Y);cuda.memcpy_htod(M_gpu, M)
    cuda.memcpy_htod(EX_gpu,EX);cuda.memcpy_htod(EX1_gpu, EX1)
    cuda.memcpy_htod(EM_gpu,EM);cuda.memcpy_htod(EM1_gpu, EM1)
    cuda.memcpy_htod(phi_n_gpu,phi_n);cuda.memcpy_htod(phi_z_gpu, phi_z)
    cuda.memcpy_htod(RR_gpu, RR);cuda.memcpy_htod(PSI_gpu,PSI)
    cuda.memcpy_htod(B11_gpu,B11);cuda.memcpy_htod(B12_gpu, B12)
    cuda.memcpy_htod(B13_gpu,B13);cuda.memcpy_htod(B14_gpu, B14)
    cuda.memcpy_htod(B21_gpu,B21);cuda.memcpy_htod(B21_gpu, B21)
    cuda.memcpy_htod(B22_gpu,B22);cuda.memcpy_htod(B23_gpu, B12);cuda.memcpy_htod(R_gpu, R)
    cuda.memcpy_htod(Q0_gpu,Q0);cuda.memcpy_htod(Q1_gpu,Q1)
    cuda.memcpy_htod(G0_gpu, G0);cuda.memcpy_htod(G1_gpu, G1)


    return (U,Y,M,EPS,EX,EM,EX1,EM1,phi_n,phi_z,RR,R,B11,B12,B13,B14,B21,B22,B23,PSI,
            U_gpu,Y_gpu,M_gpu,EPS_gpu,
            EX_gpu,EM_gpu,EX1_gpu,EM1_gpu,phi_n_gpu
            ,phi_z_gpu,RR_gpu,R_gpu,B11_gpu,B12_gpu,B13_gpu,B14_gpu,B21_gpu,B22_gpu,B23_gpu,PSI_gpu,
            Q0_gpu,Q1_gpu,G1_gpu,G0_gpu)


# In[13]:


J = 3
(U,Y,M,EPS,EX,EM,EX1,EM1,phi_n,phi_z,RR,R,B11,B12,B13,B14,B21,B22,B23,PSI,
            U_gpu,Y_gpu,M_gpu,EPS_gpu,
            EX_gpu,EM_gpu,EX1_gpu,EM1_gpu,phi_n_gpu
            ,phi_z_gpu,RR_gpu,R_gpu,B11_gpu,B12_gpu,B13_gpu,B14_gpu,B21_gpu,B22_gpu,B23_gpu,PSI_gpu,
            Q0_gpu,Q1_gpu,G1_gpu,G0_gpu)=init_gpu(para,aa,J)


# In[ ]:





# In[22]:


mod= SourceModule("""
#include <curand_kernel.h>

extern "C" {
__global__ void mult(unsigned long seed, float *U, float *R,float *EPS,float *PSI
        ,float *EX,float *EM,float *phi_n,float *phi_z,float *M ){
    int tx = threadIdx.x;
    int x= blockIdx.y*gridDim.x+blockIdx.x; 
    int tid = gridDim.y*gridDim.x;
    int a1=0;int a2=1;int a3=2;int a4=3;int a5=4;
    int b1=40;int b2=41;int b3=42;int b4=43;int b5=44;int b6=44;int b7=44;int b8=44;
    int b9=44;int b10=45;int b11=46;int b12=47;
    curandState state;
    U[x] = 1;
    U[x+1*tid] =2;
    U[x+2*tid] =3;

    // Propogate Shocks
    for(int i=0;i<5;i++){
        curand_init ( seed+i, tx, 0, &state);
        float ranv = curand_normal( &state );
        EPS[x+i*tid] = ranv;
        U[x+i*tid] =R[i]*U[x+i*tid]+PSI[i]*EPS[x+i*tid];
    }
    // Generate Expectations
    // State Variables
    for(int i=0;i<13;i++){
    EM[x+i*tid] = (phi_n[x+a1*tid]*U[x+0*tid]+
                   phi_n[x+a2*tid]*U[x+1*tid]+
                   phi_n[x+a3*tid]*U[x+2*tid]+
                   phi_n[x+a4*tid]*U[x+3*tid]+
                   phi_n[x+a5*tid]*U[x+4*tid]+
                   
                   phi_n[x+b1*tid]*M[x+0*tid]+
                   phi_n[x+b2*tid]*M[x+1*tid]+
                   phi_n[x+b3*tid]*M[x+2*tid]+
                   phi_n[x+b4*tid]*M[x+3*tid]+
                   phi_n[x+b5*tid]*M[x+4*tid]+
                   phi_n[x+b6*tid]*M[x+5*tid]+
                   phi_n[x+b7*tid]*M[x+6*tid]+
                   phi_n[x+b8*tid]*M[x+7*tid]+
                   phi_n[x+b9*tid]*M[x+8*tid]+
                   phi_n[x+b10*tid]*M[x+9*tid]+
                   phi_n[x+b11*tid]*M[x+10*tid]+
                   phi_n[x+b12*tid]*M[x+11*tid]+
                   phi_n[x+b12*tid]*M[x+12*tid] ); 
                a1 +=i*5;a2 += i*5;a3+=i*5;a4+=i*5;a5+=i*5;
                b1+=i*13;b2+=i*13;b3+=i*13;b4+=i*13;b5+=i*13;b6+=i*13;
                b7+=i*13;b8+=i*13;b9+=i*13;b10+=i*13;b11+=i*13;b12+=i*13;
    }
  a1 =64;a2 = 65;a3=66;a4=67;a5=68;
  b1=a1+20;b2+=i*13;b3+=i*13;b4+=i*13;b5+=i*13;b6+=i*13;
  b7+=i*13;b8+=i*13;b9+=i*13;b10+=i*13;b11+=i*13;b12+=i*13;
  for(int i=0;i<4;i++){
   EY[x+i*tid] =(  phi_n[x+a1*tid]*U[x+0*tid]+
                   phi_n[x+a2*tid]*U[x+1*tid]+
                   phi_n[x+a3*tid]*U[x+2*tid]+
                   phi_n[x+a4*tid]*U[x+3*tid]+
                   phi_n[x+a5*tid]*U[x+4*tid]+
                   phi_n[x+b1*tid]*M[x+0*tid]+
                   phi_n[x+b2*tid]*M[x+1*tid]+
                   phi_n[x+b3*tid]*M[x+2*tid]+
                   phi_n[x+b4*tid]*M[x+3*tid]+
                   phi_n[x+b5*tid]*M[x+4*tid]+
                   phi_n[x+b6*tid]*M[x+5*tid]+
                   phi_n[x+b7*tid]*M[x+6*tid]+
                   phi_n[x+b8*tid]*M[x+7*tid]+
                   phi_n[x+b9*tid]*M[x+8*tid]+
                   phi_n[x+b10*tid]*M[x+9*tid]+
                   phi_n[x+b11*tid]*M[x+10*tid]+
                   phi_n[x+b12*tid]*M[x+11*tid]+
                   phi_n[x+b12*tid]*M[x+12*tid] );
                   
                a1 +=i*5;a2 += i*5;a3+=i*5;a4+=i*5;a5+=i*5;
                b1+=i*13;b2+=i*13;b3+=i*13;b4+=i*13;b5+=i*13;b6+=i*13;
                b7+=i*13;b8+=i*13;b9+=i*13;b10+=i*13;b11+=i*13;b12+=i*13;
                }
    phi_n[x+40*tid] = 699;
}
}
""",no_extern_c= True)


# In[20]:


J = 3
n_phi = 156
mult = mod.get_function("mult");
BLOCK_1 = (156,1,1)
GRID_1 = (J,1,1)
sd= np.array(4321,dtype = np.float32)
sd_gpu = cuda.mem_alloc(sd.nbytes)
mult(sd,U_gpu,R_gpu,EPS_gpu,PSI_gpu,EX_gpu,EM_gpu,phi_n_gpu,phi_z_gpu,M_gpu,
     block=BLOCK_1, grid=GRID_1)
cuda.memcpy_dtoh(U, U_gpu);cuda.memcpy_dtoh(EPS, EPS_gpu);cuda.memcpy_dtoh(R, R_gpu);
cuda.memcpy_dtoh(PSI, PSI_gpu);cuda.memcpy_dtoh(EX, EX_gpu)
cuda.memcpy_dtoh(phi_z, phi_z_gpu);cuda.memcpy_dtoh(phi_n, phi_n_gpu);cuda.memcpy_dtoh(EM, EM_gpu)


# In[ ]:





# In[ ]:


phi_n


# In[ ]:


156**.5


# In[ ]:





# In[ ]:


g = 0.03
p_r_star = .1
Q0 = C11*aa;Q1 = C12*aa
G0 = A11*aa;G1 = A12*aa

Q0_z = C11_z*aa;Q1_z = C12_z*aa
G0_z = A11_z*aa;G1_z = A12_z*aa

PHI = cp.vstack([resh(Q0,[-1,1]),resh(Q1,[-1,1]),resh(G1,[-1,1]),resh(G0,[-1,1])]);
PHI = cp.vstack([resh(Q0_z,[-1,1]),resh(Q1_z,[-1,1]),resh(G1_z,[-1,1]),resh(G0_z,[-1,1])]);

RR = 1e-2*cp.eye(156)
t = 0;T = 400;FE = cp.zeros([12,T])
zx = 1
Y = zx*cp.random.standard_normal([4,T]);r_s = zx*cp.random.standard_normal([1,T])
M = zx*cp.random.standard_normal([8,T]);U = zx*cp.random.standard_normal([5,T]);
eps = zx*cp.random.standard_normal([5,T])
#eps_rs = 0*cp.random.standard_normal([1])
eps[s_sh,t] = 1
while t<T:
    #r_s[:,t] = 1*r_s[:,t-1]+eps_rs*0 
    U[:,t] = resh(m(R,U[:,t-1]),[5])+m(PSI_tilde,eps[:,t])
    #U[0,t] +=resh(((p_r_star)*r_s[:,t]),[])
    EM = m(Q0,U[:,t-1])+m(Q1,M[:,t-1])
    EM1 = m(m(Q0,R),U[:,t])+m(Q1,EM)
    EY = m(G0,U[:,t])+m(G1,EM)
    EY1 = m(m(G0,R),U[:,t])+m(G1,EM1)
    M[:,t] = m(B21,EY1)+m(B22,M[:,t-1])+m(B23,U[:,t])
    if M[7,t] <0:
        M[7,t] = 0
    Y[:,t] = m(B11,EY1)+m(B12,M[:,t])+m(B13,U[:,t])+m(B14,EM1)
    Z = cp.zeros([12,156])
    Z[0:8,0:40]= cp.kron(cp.eye(8),resh(U[:,t],[1,5]))
    Z[0:8,40:(40+64)]= cp.kron(cp.eye(8),resh(EM,[1,8]))
    Z[8:,104:(104+32)] = cp.kron(cp.eye(4),resh(EM,[1,8]))
    Z[8:,(136):(156)] = cp.kron(cp.eye(4),resh(U[:,t],[1,5]))
    y_obs = cp.vstack([resh(M[:,t],[-1,1]),resh(Y[:,t],[-1,1])])   
    EZ = m(Z,PHI)
    ERR = (EZ-y_obs)
    FE[:,t] = resh(ERR,[12]) 
    RR = (RR + g*(m(Z.T,Z)-RR))
    PHI = PHI+ g*m(m(inv(RR),Z.T),resh(ERR,[12,1]))
    Q0  = resh(PHI[0:40],[8,5])
    Q1 = resh(PHI[40:(104)],[8,8])
    G1 = resh(PHI[(104):(104+32)],[4,8])
    G0 = resh(PHI[(136):],[4,5])

    t+=1
    
plt.plot(cp.asnumpy(Y[0,20:]));
plt.plot(cp.asnumpy(M[7,20:]));
fig1, ax1, = plt.subplots() 
ax1.plot(np.arange(T),  cp.asnumpy(U[s_sh]))
ax1.set_title("shock")

fig2, ax2= plt.subplots()
ax2.plot(np.arange(T), cp.asnumpy(M[7]))
ax2.set_title("FFR")
ax2.set_xlabel("Time")
fig3, ax3 = plt.subplots() 
ax3.plot(np.arange(T),  cp.asnumpy(Y[0]))
ax3.set_title("Output")
fig4, ax4 = plt.subplots() 
ax4.plot(np.arange(T),  cp.asnumpy(Y[1]))
ax4.set_title("Inflation")
ax4.set_xlabel("Time")
fig5, ax5 = plt.subplots() 
ax5.plot(np.arange(T),  cp.asnumpy(Y[2]))
ax5.set_title("gov bond price")
ax5.set_xlabel("Time")
fig6, ax6 = plt.subplots() 
ax6.plot(np.arange(T),  cp.asnumpy(Y[3]))
ax6.set_title("corporate bond price")
ax6.set_xlabel("Time")

