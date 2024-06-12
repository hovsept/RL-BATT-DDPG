#Evaluation Code
#Hovsep - Apr 10th 2024
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.optimize
import torch
from functools import *
import time
import heapq

from ddpg_agent import Agent
from scenario_epsilon import eps_general

import gym
import scipy.signal

from gym_dfn.envs.dfn_env import *
from gym_dfn.envs.ParamFile_LCO2 import p

from settings_file import*
from collections import Counter

def get_output_observations(bat):
    return bat.SOCn, bat.V, bat.Temp

print('-'*80)
print('CC-CV Charging Protocol')
print('-'*80)

env = DFN(sett=settings, cont_sett=control_settings, param_unc=False)
plots = False
num_sims = 100
sim_saves = []
for _ in tqdm(range(num_sims)):
    try:
        init_T= np.random.uniform(low = 290, high=290)
        init_V = 2.7

        _ = env.reset(init_v = init_V, init_t=init_T)
        soc, voltage, temperature = get_output_observations(env)

        ACTION_VEC=[]
        SOC_VEC=[soc]
        T_VEC=[temperature]
        VOLTAGE_VEC=[voltage]
        RETURN_VALUE=0
        TIME_VEC = []
        done=False

        cex_sim = []
        etasLn_sim = []
        cssn_sim = []
        cssp_sim = []
        cavgn_sim = []
        cavgp_sim = []
        nLis_sim = []

        phi_s_n_sim = []
        phi_s_p_sim = []
        ien_sim = []
        iep_sim = []
        jn_sim =[]
        jp_sim = []
        j_sr_sim = []
        phi_e_sim = []

        tt=0
        TIME_VEC.append(tt)

        H = 120

        is_CC = True
        int_v = 0
        for t in range(H-1):
            if is_CC == True:
                applied_action = env.action_space.low[0]
            else:
                applied_action = a*applied_action + 2.5*(env.info['V']-4.2) + 0.1*int_v
                int_v += (env.info['V']-4.2)

            _,reward,done,_ = env.step(applied_action)
            next_soc, next_voltage, next_temperature = get_output_observations(env)

            if next_voltage>=4.16:
                if is_CC ==True and init_V>3.8:
                    applied_action = 0.65*applied_action
                is_CC = False
                if init_V <= 3.8:
                    a = 0.95
                else:
                    
                    a = 0.93
            RETURN_VALUE+=reward
            
            #save the simulation vectors
            ACTION_VEC.append(applied_action)
            SOC_VEC.append(env.info['SOCn'])
            T_VEC.append(env.info['T'])
            
            #save the simulation info
            cex_sim.append(env.c_ex)
            etasLn_sim.append(env.etasLn)
            cssn_sim.append(env.cssn)
            cssp_sim.append(env.cssp)
            cavgn_sim.append(env.c_avgn)
            cavgp_sim.append(env.c_avgp)
            nLis_sim.append(env.nLis)
            
            # save the algebraic states info.
            phi_s_n_sim.append(env.out_phisn)
            phi_s_p_sim.append(env.out_phisp)
            ien_sim.append(env.out_ien)
            iep_sim.append(env.out_iep)
            jn_sim.append(env.out_jn)
            jp_sim.append(env.out_jp)
            j_sr_sim.append(env.out_j_sr)
            phi_e_sim.append(env.out_phie)

            tt += env.dt
            if init_V == 2.7:
                tt_eval = tt
            TIME_VEC.append(tt)
            VOLTAGE_VEC.append(env.info['V'])
            # norm_out=norm_next_out
            
            if next_soc>=0.9:
                break

        sim_saves.append(np.array((TIME_VEC[-1]/60, env.Q_loss)))

        if plots == True:
            fig, ax1 = plt.subplots(figsize=(16,9))
            ax1.set_facecolor("white")
            ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.1)

            volt_th = control_settings['constraints']['voltage']['max']
            plt.plot(np.array(TIME_VEC[:])/60,np.array(VOLTAGE_VEC),linewidth=3)
            plt.plot(np.array(TIME_VEC[:])/60, volt_th*np.ones([len(VOLTAGE_VEC),]),'k--')
            plt.title("Voltage", fontsize=35, fontweight="bold")
            plt.ylabel('Voltage [$V$]', fontsize=30)
            plt.xlabel('Time [Min]', fontsize=30)
            plt.tick_params(labelsize=30)
            plt.show()

            fig, ax1 = plt.subplots(figsize=(16,9))
            ax1.set_facecolor("white")
            ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.1)

            plt.plot(np.array(TIME_VEC[:-1])/60,np.array(ACTION_VEC),linewidth=3)
            plt.title("Action", fontsize=35, fontweight="bold")
            plt.ylabel(r'Current [$A$]', fontsize=30)
            plt.xlabel('Time [Min]', fontsize=30)
            plt.tick_params(labelsize=30)
            plt.show()

            fig, ax1 = plt.subplots(figsize=(16,9))
            ax1.set_facecolor("white")
            ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.1)

            plt.plot(np.array(TIME_VEC[:])/60,np.array(SOC_VEC),linewidth=3)
            plt.plot(np.array(TIME_VEC[:])/60,0.9*np.ones([len(SOC_VEC),]),'k--')
            plt.title("SOC", fontsize=35, fontweight="bold")
            plt.ylabel('SOC [-]', fontsize=30)
            plt.xlabel('Time [Min]', fontsize=30)
            plt.tick_params(labelsize=30)
            plt.show()

            fig, ax1 = plt.subplots(figsize=(16,9))
            ax1.set_facecolor("white")
            ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.1)

            plt.plot(np.array(TIME_VEC[:-1])/60,np.array(etasLn_sim), linewidth=3)
            plt.title("Side-Reaction Overpotential", fontsize=35, fontweight="bold")
            plt.ylabel(r'$\eta_2$ [V]', fontsize=30)
            plt.xlabel('Time [Min]', fontsize=30)
            plt.tick_params(labelsize=30)
            plt.show()

            fig, ax1 = plt.subplots(figsize=(16,9))
            ax1.set_facecolor("white")
            ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.1)

            temp_th = control_settings['constraints']['temperature']['max']
            plt.plot(np.array(TIME_VEC[:])/60,np.array(T_VEC)-273 ,linewidth=3)
            plt.plot(np.array(TIME_VEC[:])/60,(temp_th-273)*np.ones([len(T_VEC),]),'k--')
            plt.title("Temperature", fontsize=35, fontweight="bold")
            plt.ylabel(r'Temperature [$^{\circ}$C]', fontsize=30)
            plt.xlabel('Time [Min]', fontsize=30)
            plt.tick_params(labelsize=30)
            plt.show()
    except:
        pass

sim_saves = np.array(sim_saves)
mean_time, mean_Q_loss = np.mean(sim_saves[:,0]), np.mean(sim_saves[:,1])
print('-'*80)
if mean_time < 0.5*H:
    print('Average Time to 80% from 2.7V: ', mean_time, 'min')
else:
    print('Time to 80% from 2.7V: NOT CHARGED')
print('-'*80)
# m = np.min(etasLn_sim)
# ub = p['a_s_n']*p['Area']*p['i0_sr']*np.exp(-2*p['alph']*p['Faraday']*m/(p['R']*p['T_amb']))*p['L_n']*TIME_VEC[-1]/3600

print('Average Charge Lost: ', mean_Q_loss, ' Ah')
print('-'*80)
cycles = np.floor(0.1*3.4/mean_Q_loss)
print('Lower Bound Cycles to 10% Capacity Loss:', cycles)
print('-'*80)
