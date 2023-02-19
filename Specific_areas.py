# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:49:02 2023

@author: Alan.Troncoso
"""
import numpy as np
import random
from scipy.spatial import distance_matrix
from scipy.stats import norm
import matplotlib.pyplot as plt

def vario_sph(h,a,C):
    h = np.asarray(h)
    h = np.where(h>=a,a,h)
    asd = C*((3/2)*(abs(h)/a)-(1/2)*(abs(h)/a)**3)
    return(asd)

def vario_Renaldy(h):    
    vario = [None]*3
    vario[0] = 21.03+vario_sph(h,132.5,12.41)+vario_sph(h,739.2,5.57) #LIM
    vario[1] = 17.28+vario_sph(h,105.4,14.36)+vario_sph(h,1201,21.58) #FSAP
    vario[2] = 570.1+vario_sph(h,1201,527.93) #SAP
    return(vario)

def dist_V_V(block,disc_x,disc_y):
    #pos_1 gives the centroid of the discretized sub blocks
    #pos_2 gives random values in v
    pos_1 = np.zeros(shape=(disc_x*disc_y,2))
    pos_2 = np.zeros(shape=(disc_x*disc_y,2))
    corner_block_x = np.linspace(0,block,disc_x+1)
    corner_block_y = np.linspace(0,block,disc_y+1)
    count = 0
    for index_x,i in enumerate(np.delete(corner_block_x,-1)):
        for index_y,j in enumerate(np.delete(corner_block_y,-1)):
            pos_1[count] = [(corner_block_x[index_x]+corner_block_x[index_x+1])/2,(corner_block_y[index_y]+corner_block_y[index_y+1])/2]
            pos_2[count] = [(random.uniform(corner_block_x[index_x],corner_block_x[index_x+1])),random.uniform(corner_block_y[index_y],corner_block_y[index_y+1])]
            count=count+1

    dist_matrix_var = distance_matrix(pos_1,pos_2).flatten() 
    return(dist_matrix_var,pos_1,pos_2)

def dist_v_V(block,disc_x,disc_y):
    #pos_1 gives the centroid of the block
    #pos_2 gives random values in v
    pos_1 = np.zeros(1)
    pos_2 = np.zeros(shape=(disc_x*disc_y,2))
    corner_block_x = np.linspace(0,block,disc_x+1)
    corner_block_y = np.linspace(0,block,disc_y+1)
    
    pos_1 = np.linspace(0,block,2)
    pos_1 = np.array([[(pos_1[0]+pos_1[1])/2,(pos_1[0]+pos_1[1])/2]])
    count = 0
    for index_x,i in enumerate(np.delete(corner_block_x,-1)):
        for index_y,j in enumerate(np.delete(corner_block_y,-1)):
            pos_2[count] = [(random.uniform(corner_block_x[index_x],corner_block_x[index_x+1])),random.uniform(corner_block_y[index_y],corner_block_y[index_y+1])]
            count=count+1

    dist_matrix_var = distance_matrix(pos_1,pos_2).flatten()
    return(dist_matrix_var,pos_1,pos_2)



def gamma_dist(n_runs,dist):
    b = np.zeros(n_runs)
    for i in np.arange(n_runs):
        vario = 1+vario_sph(dist,200,9)
        b[i] = np.mean(vario)
    return(np.mean(b))

def gamma_dist_renaldy(n_runs,dist):
    b = np.reshape(np.zeros(n_runs*3),[3,n_runs])
    b_aux = np.zeros(3)
    for i in np.arange(n_runs):
        vario = vario_Renaldy(dist)
        b[0] = np.mean(vario[0])
        b[1] = np.mean(vario[1])
        b[2] = np.mean(vario[2])
    b_aux[0] = np.mean(b[0])
    b_aux[1] = np.mean(b[1])
    b_aux[2] = np.mean(b[2])
    return(b_aux)



##########################################################################################################
#RUN single test
##########################################################################################################

#input

block = 50
disc_x = 20
disc_y = 20
mean = np.array([6.04,5.77,32.23])
n_runs = 10
mine_surface = np.array([95000,84000,57000])
############################################

#s_0_quarter = 1/(mean**2)*(2*gamma_dist_renaldy(n_runs,dist_v_V(block, disc_x, disc_y)[0])-gamma_dist_renaldy(n_runs, dist_V_V(block, disc_x, disc_y)[0]))*block**2
s_0_quarter = 1/(mean**2)*(2*gamma_dist_renaldy(n_runs,dist_v_V(block, disc_x, disc_y)[0])-gamma_dist_renaldy(n_runs, dist_V_V(block, disc_x, disc_y)[0]))*block**2
CoV_quarterly = np.sqrt(s_0_quarter/mine_surface)
CoV_yearly = CoV_quarterly/2
print(CoV_quarterly)
print(CoV_yearly)
d = 0.15
p = 0.10
CV_interval_quaterly = -d/norm.ppf(p/2)
CV_internal_yearly = 1/2*CV_interval_quaterly
print(CV_interval_quaterly)
print(CV_internal_yearly)

##########################################################################################################
#RUN array
##########################################################################################################

#block = np.array([25,50,75,100,150,200])
block = np.arange(25,225,25)
disc_x = 20
disc_y = 20
mean = np.array([6.04,5.77,32.23])
n_runs = 10
mine_surface = np.array([95000,84000,57000])
############################################
s_0_quarter = np.reshape(np.zeros(3*len(block)),[len(block),3])
s_0_yearly = np.reshape(np.zeros(3*len(block)),[len(block),3])
CoV_quarterly = np.reshape(np.zeros(3*len(block)),[len(block),3])
CoV_yearly = np.reshape(np.zeros(3*len(block)),[len(block),3])


for i in np.arange(len(block)):
    s_0_quarter[i] = 1/(mean**2)*(2*gamma_dist_renaldy(n_runs,dist_v_V(block[i], disc_x, disc_y)[0])-gamma_dist_renaldy(n_runs, dist_V_V(block[i], disc_x, disc_y)[0]))*block[i]**2
    CoV_quarterly[i] = np.sqrt(s_0_quarter[i]/mine_surface)
    CoV_yearly[i] = CoV_quarterly[i]/2
print(CoV_quarterly)
print(CoV_yearly)
d = 0.15
p = 0.10
CV_interval_quaterly = -d/norm.ppf(p/2)
CV_internal_yearly = 1/2*CV_interval_quaterly
print(CV_interval_quaterly)
print(CV_internal_yearly)

################################################
#Plots###################
################

plt.plot(block,CoV_quarterly[:,0],'o:b')
plt.axhline(y = 2*CV_interval_quaterly, color = 'r', linestyle = '-')
plt.axhline(y = CV_interval_quaterly, color = 'r', linestyle = '--')
plt.xlabel("Drillhole spacing")
plt.ylabel("CV")
plt.title("Evolution of the coefficients of variation for the LIM")


plt.plot(block,CoV_quarterly[:,1],'o:b')
plt.axhline(y = 2*CV_interval_quaterly, color = 'r', linestyle = '-')
plt.axhline(y = CV_interval_quaterly, color = 'r', linestyle = '--')
plt.xlabel("Drillhole spacing")
plt.ylabel("CV")
plt.title("Evolution of the coefficients of variation for the FSAP")


plt.plot(block,CoV_quarterly[:,2],'o:b')
plt.axhline(y = 2*CV_interval_quaterly, color = 'r', linestyle = '-')
plt.axhline(y = CV_interval_quaterly, color = 'r', linestyle = '--')
plt.xlabel("Drillhole spacing")
plt.ylabel("CV")
plt.title("Evolution of the coefficients of variation for the SAP")


plt.show()



