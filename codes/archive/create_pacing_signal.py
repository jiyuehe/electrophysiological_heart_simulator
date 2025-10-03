import numpy as np
import matplotlib.pyplot as plt

def execute(dt, t_final, pacing_start_time, pacing_cycle_length, model_flag):
    if model_flag == 1:
        pacing_duration = 10 # ms, do not change
        J_stim_value = 20 # pacing strength. 20 is good. 10 is not large enough if space step is 0.1 mm and time step is 0.001
    elif model_flag == 2:
        pacing_duration = 0.3 # ms, do not change
        J_stim_value = 1

    pacing_duration_time_steps = pacing_duration/dt # make sure it is n ms no matter what dt is
    pacing_starts = np.arange(pacing_start_time/dt, t_final/dt - pacing_duration_time_steps + 1, pacing_cycle_length/dt)
    pacing_ends = pacing_starts + pacing_duration_time_steps - 1

    t_step = len(np.arange(dt, t_final + dt, dt)) # time steps
    pacing_signal = np.zeros(t_step)
    for n in range(len(pacing_starts)):
        pacing_signal[int(pacing_starts[n]):int(pacing_ends[n])+1] = J_stim_value

    debug_plot = 0
    if debug_plot == 1:
        t = np.arange(dt, t_final + dt, dt)
        plt.figure()
        plt.plot(t,pacing_signal, 'b')
        plt.xlabel('Time (ms)')
        plt.title('Pacing signal')
        plt.show()

    return pacing_signal
