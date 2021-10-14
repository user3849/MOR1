# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 16:08:59 2021

@author: Scoot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numpy.random import default_rng
rng = default_rng()
#%%
S=11 # no of species
T=1200 # time horizon
R=11 # no of reactions
N=500 # no of MC runs
#%%
# species               index
# L                      0
# R                      1
# RL                     2
# RLw                    3
# alpha_GDP beta gamma   4
# alpha__GDP             5
# alpha_GTP              6
# beta gamma             7
# M2                     8
# CaOn                   9
# CaOff                  10

for run in range(1):
    for k_on,ca in np.array([[0.0058876,5],[0.00173472,5]]):#[0.00059333,0],[0.001461155,0].iu[0.0001,0],[0.00100744,0], [0.00191487,0], [0.00282231,0], [0.00372975,0], [0.00463719,0],[0.00554463,0], [0.00645206,0], [0.0073595,0],[0.001,5],[0.0001,5],[0.0020,5],[0.02202556,5], [0.01767812,5], [0.01333068,5], [0.00898324,5],[0.0046358,5],
            A=-np.ones((1,S+R+1,N)) # extra-columns to store reaction counts (Rea_1=A[:,S], Rea_2=A[:,S+1],..., Rea_R=A[:,S+R],...) and reaction time points
                    
            k1=k_on       # Rea_1:L+R->RL, Rea_2:R->RL      
            k3=0.0125      # Rea_4:RL+ABG->RL+AT+BG
            k5=0.0064       # Rea_6:AT->AD+M2 0.1 
            k4=0.011   # Rea_5:RL+M2->RLw
            k8=0.0522       # Rea_9:BG+CaOn->CaOff
            k6=0.1      # Rea_7:AD+CaOff->ABG+CaOn
            k2=0.0005      # Rea_3:RLw->R
            k7=0.00005     # Rea_8:RLw->0          
            k9=0.0047      # Rea_10:M2->0
            k10=0.0191     # L->0
            k11=0.00005    # Rea_4:R+ABG->R+AT+BG
            
            for n in range(N):
                
                A[0,0:S,n]=np.array([10,20,0,0,40-ca,0,ca,0,0,80-ca,ca]) # starting values
                A[0,S:S+R+1,n]=np.zeros((1,R+1))
                
                i=0
                
                while A[i,S+R,n]<T:
                    l=np.zeros(R)
                    #if A[i,S,n]<5:
                    l[0]=k1*A[i,0,n]*A[i,1,n]
                    #else: l[0]=0
                    l[1]=k11*A[i,1,n]*A[i,4,n]
                    l[2]=k2*A[i,3,n]
                    l[3]=k3*A[i,2,n]*A[i,4,n]
                    l[4]=k4*A[i,2,n]*A[i,8,n]
                    l[5]=k5*A[i,6,n]
                    l[6]=k6*A[i,5,n]*A[i,10,n]
                    l[7]=k7*A[i,3,n]
                    l[8]=k8*A[i,7,n]*A[i,9,n]
                    l[9]=k9*A[i,8,n]
                    l[10]=k10*A[i,0,n]
                    lam=np.cumsum(l)
                    p=rng.uniform(0,1)
                    tau=(1/lam[-1])*np.log(1/p)
                    if (i+1)>=np.size(A,axis=0):
                        A=np.vstack([A,-np.ones((1,S+R+1,N))])
                    A[i+1,S+R,n]=A[i,S+R,n]+tau
                    q=rng.uniform(0,1)
                    if (lam[0]/lam[-1])>q:
                        A[i+1,0,n]=A[i,0,n]-1
                        A[i+1,1,n]=A[i,1,n]-1
                        A[i+1,2,n]=A[i,2,n]+1
                        A[i+1,3:S,n]=A[i,3:S,n]
                        A[i+1,S+1:S+R,n]=A[i,S+1:S+R,n]
                        A[i+1,S,n]=A[i,S,n]+1
                    elif (lam[1]/lam[-1])>q:
                        A[i+1,4,n]=A[i,4,n]-1
                        A[i+1,6,n]=A[i,6,n]+1
                        A[i+1,7,n]=A[i,7,n]+1
                        A[i+1,0:4,n]=A[i,0:4,n]
                        A[i+1,5,n]=A[i,5,n]
                        A[i+1,8:S+1,n]=A[i,8:S+1,n]
                        A[i+1,S+2:S+R,n]=A[i,S+2:S+R,n]
                        A[i+1,S+1,n]=A[i,S+1,n]+1              
                    elif (lam[2]/lam[-1])>q:
                        A[i+1,0,n]=A[i,0,n]
                        A[i+1,3,n]=A[i,3,n]-1
                        A[i+1,1,n]=A[i,1,n]+1
                        A[i+1,2,n]=A[i,2,n]
                        A[i+1,4:S+2,n]=A[i,4:S+2,n]
                        A[i+1,S+3:S+R,n]=A[i,S+3:S+R,n]
                        A[i+1,S+2,n]=A[i,S+2,n]+1
                    elif (lam[3]/lam[-1])>q:
                        A[i+1,4,n]=A[i,4,n]-1
                        A[i+1,6,n]=A[i,6,n]+1
                        A[i+1,7,n]=A[i,7,n]+1
                        A[i+1,0:4,n]=A[i,0:4,n]
                        A[i+1,5,n]=A[i,5,n]
                        A[i+1,8:S+3,n]=A[i,8:S+3,n]
                        A[i+1,S+4:S+R,n]=A[i,S+4:S+R,n]
                        A[i+1,S+3,n]=A[i,S+3,n]+1
                    elif (lam[4]/lam[-1])>q:
                        A[i+1,2,n]=A[i,2,n]-1
                        A[i+1,8,n]=A[i,8,n]-1
                        A[i+1,3,n]=A[i,3,n]+1
                        A[i+1,0:2,n]=A[i,0:2,n]
                        A[i+1,4:8,n]=A[i,4:8,n]
                        A[i+1,9:S+4,n]=A[i,9:S+4,n]
                        A[i+1,S+5:S+R,n]=A[i,S+5:S+R,n]
                        A[i+1,S+4,n]=A[i,S+4,n]+1
                    elif (lam[5]/lam[-1])>q:
                        A[i+1,6,n]=A[i,6,n]-1
                        A[i+1,5,n]=A[i,5,n]+1
                        A[i+1,8,n]=A[i,8,n]+1
                        A[i+1,0:5,n]=A[i,0:5,n]
                        A[i+1,7,n]=A[i,7,n]
                        A[i+1,9:S+5,n]=A[i,9:S+5,n]
                        A[i+1,S+6:S+R,n]=A[i,S+6:S+R,n]
                        A[i+1,S+5,n]=A[i,S+5,n]+1                
                    elif (lam[6]/lam[-1])>q:
                        A[i+1,10,n]=A[i,10,n]-1
                        A[i+1,5,n]=A[i,5,n]-1
                        A[i+1,4,n]=A[i,4,n]+1
                        A[i+1,9,n]=A[i,9,n]+1
                        A[i+1,0:4,n]=A[i,0:4,n]
                        A[i+1,6:9,n]=A[i,6:9,n]
                        A[i+1,11:S+6,n]=A[i,11:6+S,n]
                        A[i+1,S+7:S+R,n]=A[i,S+7:S+R,n]
                        A[i+1,S+6,n]=A[i,S+6,n]+1
                    elif (lam[7]/lam[-1])>q:
                        A[i+1,3,n]=A[i,3,n]-1
                        A[i+1,0:3,n]=A[i,0:3,n]         
                        A[i+1,4:7+S,n]=A[i,4:7+S,n] 
                        A[i+1,S+8:S+R,n]=A[i,S+8:S+R,n]
                        A[i+1,S+7,n]=A[i,S+7,n]+1
                    elif (lam[8]/lam[-1])>q:
                        A[i+1,7,n]=A[i,7,n]-1
                        A[i+1,9,n]=A[i,9,n]-1
                        A[i+1,10,n]=A[i,10,n]+1
                        A[i+1,0:7,n]=A[i,0:7,n]
                        A[i+1,8,n]=A[i,8,n]    
                        A[i+1,11:8+S,n]=A[i,11:8+S,n]
                        A[i+1,S+9:S+R,n]=A[i,S+9:S+R,n]
                        A[i+1,S+8,n]=A[i,S+8,n]+1
                    elif (lam[9]/lam[-1])>q:
                        A[i+1,8,n]=A[i,8,n]-1
                        A[i+1,0:8,n]=A[i,0:8,n]   
                        A[i+1,9:9+S,n]=A[i,9:9+S,n]
                        A[i+1,10+S,n]=A[i,10+S,n]   
                        A[i+1,S+9,n]=A[i,S+9,n]+1
                    elif (lam[10]/lam[-1])>q:
                        A[i+1,0,n]=A[i,0,n]-1
                        A[i+1,1:10+S,n]=A[i,1:10+S,n]               
                        A[i+1,S+10,n]=A[i,S+10,n]+1
                    i=i+1
                    
            np.save(f'A:/trajk3=0.2k2=0.01k7=0.001k11=0.00005k1={k_on}500runs{run}fitted',A)

#%% calculate means, variances of 500 runs

t=np.linspace(0,T,121)
#%%
for k_on in np.array([0.0058876,0.00173472]): #0.00059333,0.001461155[0.0001,0.00100744,0.00191487,0.00282231,0.00372975,0.00463719,0.00554463,0.00645206,0.0073595],0.0028134,0.0021684,0.00017922,0.0073595,0.0001,0.002,0.001,0.02202556, 0.01767812, 0.01333068, 0.00898324,0.0046358,  0.0057292,0.0039493,0.026373,0.00028836
    D=np.load(f'A:/trajk3=0.2k2=0.01k7=0.001k11=0.00005k1={k_on}500runs0fitted.npy')    
    C=np.zeros((11,np.size(t)))
    B=np.zeros((11,np.size(t)))
    
    a=np.zeros(N)
    for k in range(11):
        for tstep in range(np.size(t)):
            for n in range(N):
                a[n]=D[max(np.where(np.logical_and(0<=D[:,22,n],D[:,22,n]<=t[tstep]))[0]),k,n]
            B[k,tstep]=np.mean(a)
            C[k,tstep]=np.var(a)   
    np.save(f'A:/Exk3=0.2k2=0.01k7=0.001k11=0.00005k1={k_on}500runsfitted',B)
    np.save(f'A:/Vark3=0.2k2=0.01k7=0.001k11=0.00005k1={k_on}500runsfitted',C)
    
#%% k_11 constant
E1=np.load('A:/Vark3=0.2k2=0.01k7=0.001k11=0k1=0.0028134500runsfitted.npy')
E2=np.load('A:/Vark3=0.2k2=0.01k7=0.001k11=0k1=0.0021684500runsfitted.npy')
E3=np.load('A:/Vark3=0.2k2=0.01k7=0.001k11=0k1=0.00017922500runsfitted.npy')
E4=np.load('A:/Vark3=0.2k2=0.01k7=0.001k11=0k1=0.0073595500runsfitted.npy')

B1=np.load('A:/Exk3=0.2k2=0.01k7=0.001k11=0k1=0.0028134500runsfitted.npy')#
B2=np.load('A:/Exk3=0.2k2=0.01k7=0.001k11=0k1=0.0021684500runsfitted.npy')
B3=np.load('A:/Exk3=0.2k2=0.01k7=0.001k11=0k1=0.00017922500runsfitted.npy')
B4=np.load('A:/Exk3=0.2k2=0.01k7=0.001k11=0k1=0.0073595500runsfitted.npy')

D1=np.zeros((S,np.size(t)))
D2=np.zeros((S,np.size(t)))
D3=np.zeros((S,np.size(t)))
D4=np.zeros((S,np.size(t)))
for k in range(S):
    for tstep in range(np.size(t)):
        D1[k,tstep]=1.965*(np.sqrt(E1[k,tstep]/500))
        D2[k,tstep]=1.965*(np.sqrt(E2[k,tstep]/500))
        D3[k,tstep]=1.965*(np.sqrt(E3[k,tstep]/500))
        D4[k,tstep]=1.965*(np.sqrt(E4[k,tstep]/500))
        
fig=plt.figure()
ax1 = fig.add_subplot(121)
plt.plot(np.linspace(0,2,2),np.zeros((2))+5,color='red')
plt.plot(np.linspace(0,2,2),np.zeros((2))+5,color='green')
plt.plot(np.linspace(0,2,2),np.zeros((2))+5,color='orange')
plt.plot(np.linspace(0,2,2),np.zeros((2))+5,color='blue')
plt.errorbar(t,B1[10,:],D1[10,:],errorevery=5,color='red')
plt.errorbar(t,B2[10,:],D2[10,:],errorevery=5,color='green')
plt.errorbar(t,B3[10,:],D3[10,:],errorevery=5,color='blue')
plt.errorbar(t,B4[10,:],D4[10,:],errorevery=5,color='orange')

plt.xlabel('Time (s)')
plt.ylabel('Percentage of closed calcium channels')

#plt.legend(('Fentanyl, pH=7.4','Fentanyl, pH=6.5','NFEPP, pH=7.4',' NFEPP, pH=6.5 '),loc='upper right')#'Fentanyl, pH=7.4','Fentanyl, pH=6.5','Fentanyl & NFEPP, pH=6.5 & pH=5.5',

plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*100/80))
ax1.yaxis.set_major_formatter(ticks_y)
# ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*10))
# ax1.xaxis.set_major_formatter(ticks_x)
# ax1.axvline(2, ls='--',color='black')


ax1 = fig.add_subplot(122)
#ax1 = fig.add_subplot(111)
# plt.plot(np.linspace(0,2,2),np.zeros((2)),color='red')
# plt.plot(np.linspace(0,2,2),np.zeros((2)),color='green')
# plt.plot(np.linspace(0,2,2),np.zeros((2)),color='orange')
# plt.plot(np.linspace(0,2,2),np.zeros((2)),color='blue')
plt.plot(t,E1[10,:],color='red')
plt.plot(t,E2[10,:],color='green')
plt.plot(t,E3[10,:],color='blue')
plt.plot(t,E4[10,:],color='orange')

plt.xlabel('Time (s)')
plt.ylabel('Variance')
plt.legend(('Fentanyl, pH=7.4','Fentanyl, pH=6.5','NFEPP, pH=7.4','NFEPP, pH=6.5'),loc='upper right')
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
# ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x*12))
# ax1.xaxis.set_major_formatter(ticks_x)
# ax1.axvline(2, ls='--',color='black')



