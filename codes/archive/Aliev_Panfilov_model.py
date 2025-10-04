# %%
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
two_folder_levels_up = os.path.abspath(os.path.join(script_dir, "..", ".."))
data_path = two_folder_levels_up + "/data/"
os.chdir(two_folder_levels_up) # change the working directory
if two_folder_levels_up not in sys.path:
    sys.path.insert(0, two_folder_levels_up) # Add the two-levels-up directory to sys.path

import numpy as np
import matplotlib.pyplot as plt
import codes
# %%

dt = 0.05 # Aliev–Panfilov
pacing_start_time = 1 # ms
pacing_cycle_length = 250 # ms
t_final = 550 # ms. NOTE: need to be at least long enough to have two pacings, or cannot show phase movie after simulation

# create the pacing signal
model_flag = 2
pacing_signal = codes.create_pacing_signal.execute(dt, t_final, pacing_start_time, pacing_cycle_length, model_flag)

debug_plot = 0
if debug_plot == 1: # plot pacing signal
    t = np.arange(dt, t_final + dt, dt)
    plt.figure()
    plt.plot(t,pacing_signal, 'b')
    plt.xlabel('Time (ms)')
    plt.title('Pacing signal')
    plt.show()

# %%
# k = 8.0
# a = 0.15
# epsilon_0 = 0.002
# mu_1 = 0.2
# mu_2 = 0.3
k = 1
a = 0.15
epsilon_0 = 0.002
mu_1 = 0.2
mu_2 = 0.3

action_potential = []
time = []
nsteps = int(t_final / dt)
u = 0.0  # action potential variable initial value
h = 0.0  # recovery variable initial value
for n in range(nsteps):
    t = n * dt
    
    # Model equations
    du = -k*u*(u - a)*(u - 1) - u*h + pacing_signal[n]
    dv = (epsilon_0 + mu_1*h/(u + mu_2)) * (-h - k*u*(u - a - 1))
    
    # Euler update
    u = u + dt * du
    h = h + dt * dv
    
    # Record
    action_potential.append(u)
    time.append(t)

# Plot results
plt.figure(figsize=(8,4))
plt.plot(time, action_potential, label="u (membrane potential)")
plt.xlabel("Time")
plt.ylabel("u")
plt.title("Aliev–Panfilov single cell action potentials")
plt.grid(True)
plt.show()
# %%
