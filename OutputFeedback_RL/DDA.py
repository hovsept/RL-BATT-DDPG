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

agent = Agent(state_size=3, action_size=1, random_seed=1)    

# Load
i_episode=1500

i_training=1
agent.actor_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_actor_'+str(i_episode)+'.pth',map_location='cpu'))
agent.critic_local.load_state_dict(torch.load('results_hov/training_results/training'+str(i_training)+'/episode'+str(i_episode)+'/checkpoint_critic_'+str(i_episode)+'.pth',map_location='cpu'))

print('-'*80)
print('Episode ', i_episode, ', Training ', i_training)
print('-'*80)

def policy_heatmap(agent, T = 300, max_current = -2.5*3.4, min_current = 0.):
    print("------------------------------------------------")
    print("Generating Heatmap of Policy with T = "+ str(T))
    print("------------------------------------------------")

    SOC_grid = np.linspace(0,1,20)
    V_grid = np.linspace(2.7,4.7,20)

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

ACTION = policy_heatmap(agent,T = 300)

def trajectory(agent, H):
    env = DFN(sett=settings, cont_sett=control_settings)

    init_V=np.random.uniform(low=2.7, high=4.1)
    init_T=np.random.uniform(low=290, high=305)
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

    for t in range(H):
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

    # traj = np.vstack((np.array(SOC_VEC), np.array(etasLn_sim)[:,0],
    #                    np.array(VOLTAGE_VEC), np.array(T_VEC)))
    traj = np.vstack((np.array(SOC_VEC), np.hstack((etasLn_sim[0],np.array(etasLn_sim)[:,0])),
                    np.array(VOLTAGE_VEC)))
    # traj = np.vstack((np.array(SOC_VEC), np.array([arr[-1] for arr in j_sr_sim]).T))
    # traj = np.vstack((np.array(SOC_VEC), np.array([arr[-1] for arr in cssn_sim]).T))

    while traj.shape[-1]<H:
        traj_x = np.zeros((traj.shape[0], traj.shape[1]+1))
        for i in range(traj.shape[0]):
           traj_x[i] = np.append(traj[i],traj[i,-1])
        traj = traj_x
            
    return traj

H = 200
N = 2000
n_vars = 3
all_trajs = np.zeros((N,n_vars,H))
all_time = np.zeros((N))

print('-----------------------------------------------------')
print('Generating N =', N, 'trajectories of length H =', H)
print('-----------------------------------------------------')

for i in tqdm(range(N)):
    check = False
    while check==False:
        try:
            all_trajs[i,:,:] = trajectory(agent,H)
            check=True
        except:
            pass

# np.save('traj_training'+str(i_training)+'_ep'+str(i_episode)+'.npy', all_trajs, allow_pickle=True)

# all_trajs = np.load('traj_training1_ep1500.npy')

min_eta_s = -0.03
volt_max = control_settings['constraints']['voltage']['max']
SOC_threshold = 0.8

def direct_partition(all_trajs, min_eta_s, SOC_threshold):
    all_trajs_part = np.empty((N, H), dtype='U4')
    for i in tqdm(range(N)):
        for j in range(H):
            # soc, eta_sr, V, T = all_trajs[i,0,j], all_trajs[i,1,j], all_trajs[i,2,j], all_trajs[i,3,j]
            soc, eta_sr, V = all_trajs[i,0,j], all_trajs[i,1,j], all_trajs[i,2,j]
            if soc<=0.5:
                soc_part = 'a'
            elif soc>0.5 and soc <= 0.65:
                soc_part = 'b'
            elif soc>0.65 and soc<=SOC_threshold:
                soc_part = 'c'
            else:
                soc_part = 'd'

            if eta_sr<= min_eta_s:
                eta_part = 'a'
            elif eta_sr > min_eta_s and eta_sr<= 0:
                eta_part = 'b'
            elif eta_sr >0 and eta_sr <= -min_eta_s:
                eta_part = 'c'
            else:
                eta_part = 'd'

            if V<=3.6:
                V_part = 'a'
            elif V>3.6 and V<=3.9:
                V_part = 'b'
            elif V>3.9 and V<=volt_max:
                V_part = 'c'
            else:
                V_part = 'd'

            all_trajs_part[i,j] = soc_part + eta_part + V_part


    return all_trajs_part

print('-----------------------------------------------------')
print('Partitioning Trajectories')
print('-----------------------------------------------------')
all_trajs_part = direct_partition(all_trajs, min_eta_s,SOC_threshold)

# unsafe_traj = []
# for i in range(N):
#     if all_trajs_part[i][0] == 0 and 1 in all_trajs_part[i]:
#         unsafe_traj.append(i)


ell = 30

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

if len(ell_seq_trajectory) > len(ell_seq_init):
    print(f'Visited ell-sequences are more than the initial ones: \n'
          f'visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}.')
elif len(ell_seq_trajectory) > len(ell_seq_init):
    print(f'Randomly picked ell-sequences == visited partitions: \n'
          f'visited {len(ell_seq_trajectory)}, initial: {len(ell_seq_init)}.')
else:
    print(f'Same number of seen and initial sequences: ({len(ell_seq_init)}).')

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


################################################
# 2D-Visualization for SOC and eta_sr trajectories
################################################

fig, ax = plt.subplots()
for seq in all_trajs[:]:
    ax.plot(seq[0][1:],seq[1][1:])
    # ax.fill_between(seq[0],-0.04,  min_eta_s, where=seq[0]>=0, color = "lightcoral")
    # ax.fill_between(seq[0],-0.04, min_eta_s, where = seq[0]>=SOC_threshold, color = "palegreen")

ax.set_xlabel('SOC')
ax.set_ylabel('eta_sr')
ax.set_title("N = " + str(N))
ax.grid()
fig.tight_layout()

################################################
# 2D-Visualization for SOC and V trajectories
################################################

fig, ax = plt.subplots()
for seq in all_trajs[:]:
    ax.plot(seq[0],seq[2])
    # ax.fill_between(seq[0],-0.04,  min_eta_s, where=seq[0]>=0, color = "lightcoral")
    # ax.fill_between(seq[0],-0.04, min_eta_s, where = seq[0]>=SOC_threshold, color = "palegreen")

ax.set_xlabel('SOC')
ax.set_ylabel('Voltage (V)')
ax.set_title("N = " + str(N))
ax.grid()
fig.tight_layout()

################################################
# 2D-Visualization for V and eta_sr trajectories
################################################

fig, ax = plt.subplots()
for seq in all_trajs[:]:
    ax.plot(seq[2][1:],seq[1][1:])
    # ax.fill_between(seq[0],-0.04,  min_eta_s, where=seq[0]>=0, color = "lightcoral")
    # ax.fill_between(seq[0],-0.04, min_eta_s, where = seq[0]>=SOC_threshold, color = "palegreen")

ax.set_xlabel('Voltage(V)')
ax.set_ylabel('eta_sr')
ax.set_title("N = " + str(N))
ax.grid()
fig.tight_layout()


################################################
# Counterexamples
################################################

# fig, ax = plt.subplots()
# for seq in all_trajs[unsafe_traj]:
#     if seq[1][-1]< seq[1][0]:
#         ax.plot(seq[0],seq[1])
#         ax.fill_between(seq[0],4.75* min_eta_s,  min_eta_s, where=seq[0]>=0, color = "lightcoral")
#         ax.fill_between(seq[0], min_eta_s,1e-6, where = seq[0]>=SOC_threshold, color = "palegreen")

# ax.set_xlabel('SOC')
# ax.set_ylabel('i_s')
# ax.set_title("Number of Counterexamples: "+ str(len(unsafe_traj)))
# fig.tight_layout()

print("Upper bound of complexity ", num_sets)

print('-'*80)
epsi_up = eps_general(k=num_sets, N=N, beta=1e-6)
print(f'Epsilon Bound using complexity: {epsi_up}')

print('-'*80)
print('Minimum Reached eta_SR: ', np.min(all_trajs[:,1]))

##################################################

env = DFN(sett=settings, cont_sett=control_settings)

init_V=2.7
init_T=np.random.uniform(low=298, high=305)
_ = env.reset(init_v = init_V, init_t=init_T)
soc, voltage, temperature = get_output_observations(env)
norm_out = normalize_outputs(soc,voltage,temperature)

TIME_VEC = []
done=False

tt=0
TIME_VEC.append(tt)

for t in range(H):
    # the exploration noise has been disabled
    norm_action = agent.act(norm_out, add_noise=False)
    

    applied_action=denormalize_input(norm_action,
                                            env.action_space.low[0],
                                            env.action_space.high[0])    

    _,reward,done,_ = env.step(applied_action)
    next_soc, next_voltage, next_temperature = get_output_observations(env)
    norm_next_out = normalize_outputs(next_soc, next_voltage, next_temperature.item())
    
    tt += env.dt
    TIME_VEC.append(tt)
    norm_out=norm_next_out
    
    if done:
        break

print('-'*80)
if TIME_VEC[-1] < H*30:
    print('Time from 2.7V: ', TIME_VEC[-1]/60, 'min')
else:
    print('Time from 2.7V: NOT CHARGED')
print('-'*80)

############################################################
# Backwards Reachability
############################################################

self_loops = set()
for traj in ell_seq_trajectory:
    if all(i==traj[0] for i in traj):
        self_loops.add(traj)

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
    pre_old = pre_state
    while True:
        for s in pre_state:
            if s not in pre_done:
                pre_old = pre_state
                pre_state = pre_state.union(pre(s,ell_seq_trajectory))
                pre_done.add(s)
        
        if pre_old == pre_state:
            break
    return pre_state

def pre_set_init(state,ell_seq_trajectory,ell_seq_init):
    pre_state = pre_set(state,ell_seq_trajectory)
    return pre_state.intersection(ell_seq_init)


soc_reach, eta_viol, volt_viol = set(), set(), set()
soc_pre, eta_pre, volt_pre = set(), set(), set()

for s in tqdm(ell_seq_trajectory):
    if s[-1][0]=='c' and s not in soc_pre:
        soc_reach.add(s)
        soc_pre = soc_pre.union(pre_set(s,ell_seq_trajectory))
    if s[-1][1]=='a' and s not in eta_pre:
        eta_viol.add(s)
        eta_pre = eta_pre.union(pre_set(s,ell_seq_trajectory))
    if s[-1][2]=='d' and s not in volt_pre:
        volt_viol.add(s)
        volt_pre = volt_pre.union(pre_set(s,ell_seq_trajectory))

soc_init = soc_pre.intersection(ell_seq_init)
eta_init = eta_pre.intersection(ell_seq_init)
volt_init = volt_pre.intersection(ell_seq_init)

eta_counterex = np.zeros((3,))
for init in eta_init:
    if init[-1][-1] == 'a':
        eta_counterex[0]+=1
    elif init[-1][-1] == 'b':
        eta_counterex[1]+=1
    elif init[-1][-1] == 'c':
        eta_counterex[2]+=1

volt_counterex = np.zeros((3,))
for init in volt_init:
    if init[-1][-1] == 'a':
        volt_counterex[0]+=1
    elif init[-1][-1] == 'b':
        volt_counterex[1]+=1
    elif init[-1][-1] == 'c':
        volt_counterex[2]+=1





