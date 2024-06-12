#Data-Driven Abstraction
#Hovsep Touloujian - Dec 8th 2023
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

def normalize_outputs(soc, voltage, temperature):

    norm_soc = soc - 0.5
    norm_voltage = (voltage - 3.5) / 1.2
    norm_temperature = (temperature - 298 - 10) / (320 - 298)
    norm_output = np.array([norm_soc, norm_voltage, norm_temperature])

    return norm_output

def get_output_observations(bat):
    return bat.SOCn, bat.V, bat.Temp

def denormalize_input(input_value, min_OUTPUT_value, max_OUTPUT_value):
    
    output_value=(1+input_value)*(max_OUTPUT_value-min_OUTPUT_value)/2+min_OUTPUT_value
    
    return output_value

switched_controller = False
agent = Agent(state_size=3, action_size=1, random_seed=1)    

# Load
load_model = False #True if generating trajectories from this code or from .npy file
if load_model == True:
    i_episode=3000

    i_training=1
    agent.actor_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_'+str(i_episode)+'.pth',map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_'+str(i_episode)+'.pth',map_location='cpu'))

    agent1 = Agent(state_size=3, action_size=1, random_seed=1)
    i_episode = 2000
    i_training = 1
    agent1.actor_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_'+str(i_episode)+'.pth',map_location='cpu'))
    agent1.critic_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_'+str(i_episode)+'.pth',map_location='cpu'))

    agent2 = Agent(state_size=3, action_size=1, random_seed=1)
    i_episode = 2560
    i_training = 2
    agent2.actor_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_'+str(i_episode)+'.pth',map_location='cpu'))
    agent2.critic_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_'+str(i_episode)+'.pth',map_location='cpu'))

    agent3 = Agent(state_size=3, action_size=1, random_seed=1)
    i_episode = 3000
    i_training = 2
    agent3.actor_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_'+str(i_episode)+'.pth',map_location='cpu'))
    agent3.critic_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_'+str(i_episode)+'.pth',map_location='cpu'))

        


    print('-'*80)
    print('Episode ', i_episode, ', Training ', i_training)
    print('-'*80)

    def policy_heatmap(agent, T = 300, max_current = -2.5*3.4, min_current = 0.):
        print("------------------------------------------------")
        print("Generating Heatmap of Policy with T = "+ str(T))
        print("------------------------------------------------")

        SOC_grid = np.linspace(0,1,20)
        V_grid = np.linspace(2.7,4.3,20)

        ACTION = np.zeros((len(SOC_grid)))

        for V in V_grid:
            ACTION_I = np.array([])
            for soc in SOC_grid:

                norm_out = normalize_outputs(soc,V,T)
                action = agent.act(norm_out, add_noise = False)
                applied_action = denormalize_input(action, max_current, min_current)
                # applied_action = action

                ACTION_I = np.hstack((ACTION_I, applied_action))

            ACTION = np.vstack((ACTION_I, ACTION))

        ACTION = ACTION[:-1]

        nx = SOC_grid.shape[0]
        no_labels_x = nx # how many labels to see on axis x
        step_x = int(nx / (no_labels_x - 1)) # step between consecutive labels
        x_positions = np.arange(0,nx,step_x) # pixel count at label position

        ny = V_grid.shape[0]
        no_labels_y = ny # how many labels to see on axis x
        step_y = int(ny / (no_labels_y - 1)) # step between consecutive labels
        y_positions = np.arange(0,ny,step_x) # pixel count at label position


        plt.imshow(ACTION, interpolation='nearest')
        plt.colorbar()
        plt.xticks(x_positions, np.trunc(100*SOC_grid)/100, rotation = -90)
        plt.yticks(y_positions, np.trunc(100*np.flip(V_grid))/100)
        plt.tight_layout()
        plt.title('RL Policy, T = ' + str(T))
        plt.xlabel('SOC')
        plt.ylabel('Voltage (V)')
        plt.show()
        return ACTION

    if switched_controller == False:
        ACTION = policy_heatmap(agent,T = 300)

    def trajectory(agent, agent1, agent2, agent3, H, switched_controller):
        env = DFN(sett=settings, cont_sett=control_settings)

        init_V=np.random.uniform(low=2.7, high=4.1)
        init_T=np.random.uniform(low=290, high=305)

        if switched_controller == True:
            if init_V <= 3.6:
                agent = agent2
            elif init_V > 3.6 and init_V <= 3.9:
                agent = agent1
            elif init_V>3.9:
                agent = agent3

        _ = env.reset(init_v = init_V, init_t=init_T)
        soc, voltage, temperature = get_output_observations(env)
        norm_out = normalize_outputs(soc,voltage,temperature)

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

        for t in range(H-1):
            # the exploration noise has been disabled
            norm_action = agent.act(norm_out, add_noise=False)
            

            applied_action=denormalize_input(norm_action,
                                                    env.action_space.low[0],
                                                    env.action_space.high[0])    

            _,reward,done,_ = env.step(applied_action)
            next_soc, next_voltage, next_temperature = get_output_observations(env)
            norm_next_out = normalize_outputs(next_soc, next_voltage, next_temperature.item())

            RETURN_VALUE+=reward
            
            #save the simulation vectors
            ACTION_VEC.append(applied_action[0])
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
            TIME_VEC.append(tt)
            VOLTAGE_VEC.append(env.info['V'])
            norm_out=norm_next_out
            
            if done:
                break
        traj = np.vstack((np.array(SOC_VEC), np.hstack((etasLn_sim[0],np.array(etasLn_sim)[:,0])),
                        np.array(VOLTAGE_VEC), np.array(T_VEC)))

        while traj.shape[-1]<H:
            traj_x = np.zeros((traj.shape[0], traj.shape[1]+1))
            for i in range(traj.shape[0]):
                traj_x[i] = np.append(traj[i],traj[i,-1])
                traj = traj_x
                
        return traj

    H = 120
    N = 30
    n_vars = 4
    all_trajs = np.zeros((N,n_vars,H))
    all_time = np.zeros((N))

    print('-----------------------------------------------------')
    print('Generating N =', N, 'trajectories of length H =', H)
    print('-----------------------------------------------------')

    for i in tqdm(range(N)):
        check = False
        while check==False:
            try:
                all_trajs[i,:,:] = trajectory(agent, agent1, agent2, agent3 , H, switched_controller)
                check=True
            except:
                pass
else:
    # all_trajs = np.load('DDA_traces/traj_training1_ep3000.npy')
    # all_trajs = np.load('DDA_traces/traj_training_CC_CV.npy')
    # all_trajs = np.vstack((np.load('DDA_traces/traj_training_counterex2_2.npy'),np.load('DDA_traces/traj_training_counterex2_5.npy'),
    #                         np.load('DDA_traces/traj_training_counterex2_6.npy')))
    all_trajs = np.load('DDA_traces/traj_training_counterex_final.npy')
    # all_trajs = np.load('DDA_traces/traj_training_counterex_extra.npy')

N = all_trajs.shape[0]
H = 120
all_trajs = all_trajs[:,:,:H]

min_eta_s = -0.2453
volt_max = control_settings['constraints']['voltage']['max']
temp_max = control_settings['constraints']['temperature']['max']
volt_max = 4.218
temp_max = 309.376
SOC_threshold = 0.8

def direct_partition(all_trajs, min_eta_s, SOC_threshold, volt_max):
    N, H = all_trajs.shape[0], all_trajs.shape[-1]
    all_trajs_part = np.empty((N, H), dtype='U4')
    for i in tqdm(range(N)):
        for j in range(H):
            soc, eta_sr, V, T = all_trajs[i,0,j], all_trajs[i,1,j], all_trajs[i,2,j], all_trajs[i,3,j]
            
            if soc<=0.1:
                soc_part = 'a'
            elif soc>0.1 and soc<=0.2:
                soc_part = 'b'
            elif soc>0.2 and soc<=0.3:
                soc_part = 'c'
            elif soc>0.3 and soc<=0.4:
                soc_part = 'd'
            elif soc>0.4 and soc<=0.45:
                soc_part = 'e'
            elif soc> 0.45 and soc<=0.5:
                soc_part = 'f'
            elif soc>0.5 and soc<=0.525:
                soc_part = 'g'
            elif soc>0.525 and soc<=0.55:
                soc_part = 'h'
            elif soc>0.55 and soc<=0.575:
                soc_part = 'i'
            elif soc>0.575 and soc<=0.6:
                soc_part = 'j'
            elif soc> 0.6 and soc <= 0.625:
                soc_part = 'k'
            elif soc> 0.625 and soc <= 0.65:
                soc_part = 'l'
            elif soc>0.65 and soc<=0.675:
                soc_part = 'm'
            elif soc>0.675 and soc<=0.7:
                soc_part = 'n'
            elif soc>0.7 and soc<=0.725:
                soc_part = 'o'
            elif soc>0.725 and soc<= 0.75:
                soc_part = 'p'
            elif soc>0.75 and soc<= 0.775:
                soc_part = 'q'
            elif soc>0.775 and soc<= 0.7875:
                soc_part = 'r'
            elif soc>0.775 and soc<= SOC_threshold:
                soc_part = 's'
            else:
                soc_part = 't'

            if eta_sr >= min_eta_s:
                eta_part = 'a'
            else:
                eta_part = 'b'

            if V<=volt_max:
                V_part = 'a'
            else:
                V_part = 'b'

            if T<= 35+273:
                T_part = 'a'
            else:
                T_part = 'b'

            all_trajs_part[i,j] = soc_part + eta_part + V_part + T_part


    return all_trajs_part

print('-----------------------------------------------------')
print('Partitioning Trajectories')
print('-----------------------------------------------------')
all_trajs_part = direct_partition(all_trajs, min_eta_s,SOC_threshold, volt_max)


ell = 12

def get_ell_sequences(all_trajs_part, ell,H):
    ell_seq_trajectory = set()
    ell_seq_init = set()
    for trajectory_parts in tqdm(all_trajs_part):
        idx = 0
        for idx in range(0, H-ell+1):
            ell_seq_trajectory.add( tuple(trajectory_parts[idx:idx+ell]) )
        # find all ell-seq from INITIAL STATE
        ell_seq_init.add(tuple(trajectory_parts[0:ell]))
        # find ONE ell-seq from a trajectory at a random point

    return ell_seq_trajectory, ell_seq_init

print('-----------------------------------------------------')
print('Generating ell-Sequences ell = ', str(ell))
print('-----------------------------------------------------')

ell_seq_trajectory, ell_seq_init = get_ell_sequences(all_trajs_part, ell, H)
self_loops = set()
for traj in ell_seq_trajectory:
    if all(i==traj[0] for i in traj):
        self_loops.add(traj)

if len(ell_seq_trajectory) > len(ell_seq_init):
    print(f'Visited ell-sequences are more than the initial ones: \n'
          f'visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}.')
elif len(ell_seq_trajectory) > len(ell_seq_init):
    print(f'Randomly picked ell-sequences == visited partitions: \n'
          f'visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}.')
else:
    print(f'Same number of seen and initial sequences: ({len(ell_seq_init)}).')

################################################
# Computing Upper Bound of Complexity
################################################

def greedy_set_cover(subsets: set, parent_set: set):
    #parent_set = set(e for s in parent_set for e in s)
    max = len(parent_set)
    # create the initial heap.
    # Note 'subsets' can be unsorted,
    # so this is independent of whether remove_redunant_subsets is used.
    heap = []
    for s in subsets:
        # Python's heapq lets you pop the *smallest* value, so we
        # want to use max-len(s) as a score, not len(s).
        # len(heap) is just proving a unique number to each subset,
        # used to tiebreak equal scores.
        heapq.heappush(heap, [max-len(s), len(heap), s])
    #results = []
    result_set = set()
    num_sets = 0
    u = 1
    tic = time.perf_counter()
    while result_set < parent_set:
        #logging.debug('len of result_set is {0}'.format(len(result_set)))
        best = []
        unused = []
        while heap:
            score, count, s = heapq.heappop(heap)
            if not best:
                best = [max-len(s - result_set), count, s]
                continue
            if score >= best[0]:
                # because subset scores only get worse as the resultset
                # gets bigger, we know that the rest of the heap cannot beat
                # the best score. So push the subset back on the heap, and
                # stop this iteration.
                heapq.heappush(heap, [score, count, s])
                break
            score = max-len(s - result_set)
            if score >= best[0]:
                unused.append([score, count, s])
            else:
                unused.append(best)
                best = [score, count, s]
        add_set = best[2]
        #logging.debug('len of add_set is {0} score was {1}'.format(len(add_set), best[0]))
        #results.append(add_set)
        result_set.update(add_set)
        num_sets += 1
        # subsets that were not the best get put back on the heap for next time.
        while unused:
            heapq.heappush(heap, unused.pop())
        if (len(result_set) / (u*2000)) > 1:
            toc = time.perf_counter()
            # Print percentage of covered elements
            print(f'{len(result_set)/len(parent_set)*100:.2f}%')
            u += 1
            print(f'Elapsed time: {toc - tic:0.4f} seconds')
            tic = toc
    return num_sets

num_sets = 0
if ell < H:
    # Recast for set cover problem
    subsets = []
    for H_seq in all_trajs_part:
        seq_of_ell_seq = []
        for i in range(0, len(H_seq)-ell+1):
            seq_of_ell_seq.append(tuple(H_seq[i:i+ell]))
        subsets.append(set(seq_of_ell_seq))
    tic = time.perf_counter()
    num_sets = greedy_set_cover(subsets, ell_seq_trajectory)
    toc = time.perf_counter()
    print(f"Time elapsed: {toc - tic:0.4f} seconds")
else:
    num_sets = len(ell_seq_trajectory)

##################################################

print("Upper bound of complexity ", num_sets)

print('-'*80)
epsi_up = eps_general(k=num_sets, N=N, beta=1e-6)
print(f'Epsilon Bound using complexity: {epsi_up}')

print('-'*80)
print('Minimum Reached eta_SR: ', np.min(all_trajs[:,1]))

print('-'*80)
print('Maximum Reached Voltage: ', np.max(all_trajs[:,2]))

print('-'*80)
print('Maximum Reached Temperature: ', np.max(all_trajs[:,3]))

################################################
# 2D-Visualization for SOC and eta_sr trajectories
################################################

fig, ax = plt.subplots()
for seq in all_trajs[:]:
    ax.plot(seq[0][1:],seq[1][1:])
    ax.fill_between(seq[0],-0.3,  min_eta_s, where=seq[0]>=0, color = "lightcoral")
    ax.fill_between(seq[0], min_eta_s, np.max(all_trajs[:,1]), where = seq[0]>=SOC_threshold, color = "palegreen")

ax.set_xlabel('SOC')
ax.set_ylabel(r'$\eta_2 [V]$')
ax.set_title("N = " + str(N))
ax.grid()
fig.tight_layout()

################################################
# 2D-Visualization for SOC and V trajectories
################################################

fig, ax = plt.subplots()
for seq in all_trajs[:]:
    ax.plot(seq[0],seq[2])
    ax.fill_between(seq[0],volt_max,  4.35, where=seq[0]>=0, color = "lightcoral")
    ax.fill_between(seq[0],2.7, volt_max, where = seq[0]>=SOC_threshold, color = "palegreen")

ax.set_xlabel('SOC')
ax.set_ylabel('Voltage (V)')
ax.set_title("N = " + str(N))
ax.grid()
fig.tight_layout()

################################################
# 2D-Visualization for temp and SOC trajectories
################################################

fig, ax = plt.subplots()
for seq in all_trajs[:]:
    ax.plot(seq[0],seq[3])
    ax.fill_between(np.linspace(0,0.9,10),temp_max, temp_max +5, where= np.linspace(0,0.9,10)>=0, color = "lightcoral")
    ax.fill_between(np.linspace(0.8,0.9,10),290, 35+273, where = np.linspace(0.8,0.9,10)>=0.8, color = "palegreen")

ax.set_xlabel('SOC')
ax.set_ylabel('Temperature (K)')
ax.set_title("N = " + str(N))
ax.grid()
fig.tight_layout()


################################################
# Charging Protocol Evaluation
################################################

i_episode=2000
i_training=2

agent.actor_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_'+str(i_episode)+'.pth',map_location='cpu'))
agent.critic_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_'+str(i_episode)+'.pth',map_location='cpu'))
F = 0.3

env = DFN(sett=settings, cont_sett=control_settings, param_unc=False)

num_sims = 1
sim_saves = []
for _ in tqdm(range(num_sims)):
    try:
        init_T= np.random.uniform(low = 290, high=290)
        init_V = 2.7

        _ = env.reset(init_v = init_V, init_t=init_T)
        soc, voltage, temperature = get_output_observations(env)
        norm_out = normalize_outputs(soc,voltage,temperature)

        ACTION_VEC=[]
        SOC_VEC=[soc]
        T_VEC=[temperature]
        VOLTAGE_VEC=[voltage]
        RETURN_VALUE=0
        TIME_VEC = []
        done=False
        tt=0
        TIME_VEC.append(tt)

        H = 120

        is_CC = True
        int_v = 0
        for t in range(H-1):
        # the exploration noise has been disabled
            norm_action = agent.act(norm_out, add_noise=False)
            

            applied_action=denormalize_input(norm_action,
                                                    env.action_space.low[0],
                                                    env.action_space.high[0])

            if t>=1:
                applied_action = F*ACTION_VEC[-1] + (1-F)*applied_action
            
            ACTION_VEC.append(applied_action)

            _,reward,done,_ = env.step(applied_action)
            next_soc, next_voltage, next_temperature = get_output_observations(env)
            if next_soc >=SOC_threshold:
                done = True

            norm_next_out = normalize_outputs(next_soc, next_voltage, next_temperature.item())
            
            tt += env.dt
            TIME_VEC.append(tt)
            norm_out=norm_next_out
        
            if done == True or next_soc>=0.9:
                break
    except:
        pass
    
    sim_saves.append(np.array((TIME_VEC[-1]/60, env.Q_loss)))

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
print('Lower Bound Cycles to 10% Capacity Loss:', np.floor(0.1*3.4/mean_Q_loss))
print('-'*80)

############################################################
# Backwards Reachability
############################################################

def pre(state, ell_seq_trajectory):
    #Returns set of ell-sequences that transition to state in one step
    ell = len(state)
    pre_state = set()
    for s in ell_seq_trajectory:
        if s[1:] == state[:ell-1]:
            pre_state.add(s)
    return pre_state

def pre_set(state, ell_seq_trajectory):
    #Returns set of ell-sequences that transition to state in any number of steps
    pre_state = pre(state,ell_seq_trajectory)
    pre_done = {state}
    # pre_old = pre_state
    while True:
        for s in pre_state:
            if s not in pre_done:
                # pre_old = pre_state
                pre_state = pre_state.union(pre(s,ell_seq_trajectory - pre_state))
                pre_done.add(s)
        
        if pre_state.union({state}) == pre_done:
            break
    return pre_state

def pre_set_init(state,ell_seq_trajectory,ell_seq_init):
    pre_state = pre_set(state,ell_seq_trajectory)
    return pre_state.intersection(ell_seq_init)


soc_reach, eta_viol, volt_viol, temp_viol = set(), set(), set(), set()
soc_pre, eta_pre, volt_pre, temp_pre = set(), set(), set(), set()

print('-'*80)
print('Backwards Reachability')
print('-'*80)

for s in tqdm(ell_seq_trajectory):
    if s[-1][0]=='s' and s not in soc_pre:
        soc_reach.add(s)
        soc_pre = soc_pre.union(pre_set(s,ell_seq_trajectory))
    if s[-1][1]=='b' and s not in eta_pre:
        eta_viol.add(s)
        eta_pre = eta_pre.union(pre_set(s,ell_seq_trajectory))
    if s[-1][2]=='b' and s not in volt_pre:
        volt_viol.add(s)
        volt_pre = volt_pre.union(pre_set(s,ell_seq_trajectory))
    if s[-1][3] == 'b' and s not in temp_pre:
        temp_viol.add(s)
        temp_pre = temp_pre.union(pre_set(s,ell_seq_trajectory))

soc_init = soc_pre.intersection(ell_seq_init)
eta_init = eta_pre.intersection(ell_seq_init)
volt_init = volt_pre.intersection(ell_seq_init)
temp_init = temp_pre.intersection(ell_seq_init)

####################
# Directly Obtaining initial conditions 
# from counterexample trajectories
####################

volt_ic = []
eta_ic = []
temp_ic = []
num_viol = 0

for seq in all_trajs[:]:
    viol_seq = 0
    if any(v>=4.2 for v in seq[2]):
        volt_ic.append((seq[2][0],seq[3][0]))
        viol_seq = 1
    if any(eta<=min_eta_s for eta in seq[1]):
        eta_ic.append((seq[2][0],seq[3][0]))
        viol_seq = 1
    if any(T>=35+273 for T in seq[3]):
        temp_ic.append((seq[2][0],seq[3][0]))
        viol_seq = 1
    if viol_seq == 1:
        num_viol += 1

volt_ic = np.array(volt_ic)
eta_ic = np.array(eta_ic)
temp_ic = np.array(temp_ic)

print('Percentage of Violations: ', 100*num_viol/N, '%')

plt.grid()
plt.scatter(volt_ic[:,0],volt_ic[:,1])
plt.xlim((2.7,4.1))
plt.ylim((290,305))
plt.xlabel(r'$V_0 (V)$')
plt.ylabel(r'$T_0(K)$')
plt.title('Voltage Safety Violations')
plt.show()


plt.grid()
plt.scatter(temp_ic[:,0],temp_ic[:,1])
plt.xlim((2.7,4.1))
plt.ylim((290,305))
plt.xlabel(r'$V_0 (V)$')
plt.ylabel(r'$T_0(K)$')
plt.title('Temperature Safety Violations')
plt.show()