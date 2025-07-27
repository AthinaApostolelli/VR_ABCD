import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os, re
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks
import pandas as pd
import yaml
import math
from math import log10, floor
import itertools
import seaborn as sns


def compute_deltaF_F0(basepath, animal, session, valid_frames, reload=False, method='Sandra', funcimg_frame_rate=45):
    DF_F_file = os.path.join(basepath, animal, session, 'funcimg/Session/suite2p/plane0/DF_F0.npy')

    if os.path.exists(DF_F_file) and reload is False:
        print('DF_F0 file found. Loading...')
        
        DF_F_all = np.load(DF_F_file)
        DF_F = DF_F_all[:, valid_frames]
        print(DF_F.shape)
        
    else:
        reload = True
        F = np.load(os.path.join(basepath, animal, session, 'funcimg/Session/suite2p/plane0/F.npy'))
        Fneu = np.load(os.path.join(basepath, animal, session, 'funcimg/Session/suite2p/plane0/Fneu.npy'))

        # Load suite2p labels and filter valid neurons
        iscell = np.load(os.path.join(basepath, animal, session, 'funcimg/Session/suite2p/plane0/iscell.npy'))[:,0]
        neurons = np.where(iscell == 1)[0] 

        # Calculate deltaF/F0
        if method == 'Sandra':
            # Option 1 - Sandra: using moving percentile for F0 (https://www.nature.com/articles/s41586-021-03452-z#Sec7)
            # F0 is defined as the 25th percentile of the fluorescence trace in a sliding window of 60 s
            # The average green fluorescence signal was extracted for each cell and then corrected for neuropil contamination 
            # by subtracting the signal of 30 μm surrounding each cell multiplied by 0.7 and adding the median multiplied by 0.7

            Fcorr = F - 0.7 * Fneu + 0.7 * np.median(Fneu, axis=1).reshape(-1,1)

            F0 = np.zeros(np.shape(F))
            f0_window = 60 * funcimg_frame_rate  # frames
            for n in neurons:  # Loop over neurons (rows)
                F0[n, :] = percentile_filter(F[n, :], percentile=25, size=f0_window, mode='nearest')

            DF_F_all = (Fcorr - F0) / F0  # Compute DF/F as (F-F0)/F0 per frame per neuron

            # Select the correct frames that fall within VR behaviour 
            DF_F = DF_F_all[:, valid_frames]

            for n in neurons[0:10]:
                plt.figure()
                plt.plot(F[n,0:2000], label='F')
                plt.plot(Fcorr[n,0:2000], label='Fcorr')
                plt.plot(F0[n,0:2000], label='F0')
                plt.plot(DF_F[n,0:2000], label='DF/F0')
                plt.legend()


def get_psth(data, neurons, event_idx, time_around=(-1, 3), funcimg_frame_rate=45):
    num_neurons = len(neurons)

    # Handle single int input as symmetric window
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a single number or a tuple/list of (start, end)")

    start_frames = int(np.floor(start_time * funcimg_frame_rate))
    end_frames = int(np.ceil(end_time * funcimg_frame_rate))
    time_bins = end_frames - start_frames

    # Get indices for each event
    window = np.arange(start_frames, end_frames)
    window_indices = np.add.outer(event_idx, window).astype(int)

    # Remove last events if close to session end 
    valid_mask = window_indices[:, -1] < data.shape[1]
    valid_window_indices = window_indices[valid_mask]

    # Preallocate PSTH array
    num_events = valid_window_indices.shape[0]
    psth = np.zeros((num_neurons, num_events, time_bins))
    for n, neuron in enumerate(neurons):
        psth[n, :, :] = data[neuron, valid_window_indices]

    # Compute average PSTH across events
    average_psth = np.mean(psth, axis=1)

    return psth, average_psth


def plot_avg_psth(average_psth, event='reward', zscoring=True, time_around=(-1, 1), funcimg_frame_rate=45, save_psth=False, savepath='', filename=''):

    # Handle time window
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a number or a tuple/list (start, end)")

    start_frames = int(np.floor(start_time * funcimg_frame_rate))
    end_frames = int(np.ceil(end_time * funcimg_frame_rate))
    num_timebins = average_psth.shape[1]
    num_neurons = average_psth.shape[0]

    # Index corresponding to event time (0s)
    zero_bin = int(round(-start_frames))

    # Sort cells by time of max response
    sortidx = np.argsort(np.argmax(average_psth, axis=1))

    data = average_psth.copy()
    if zscoring:
        data = stats.zscore(data, axis=1)

    fig, ax = plt.subplots(figsize=(3, 4))
    im = ax.imshow(data[sortidx, :], aspect='auto')

    # Event marker line (time 0)
    ax.vlines(zero_bin - 0.5, ymin=-0.5, ymax=num_neurons - 0.5, color='k')

    ax.set_xlabel('Time (s)')
    ax.set_xticks([0, zero_bin, num_timebins - 1])
    ax.set_xticklabels([round(start_time, 2), 0, round(end_time, 2)])

    ax.set_ylabel('Neuron')
    ax.set_yticks([-0.5, num_neurons - 0.5])
    ax.set_yticklabels([0, num_neurons])
    fig.suptitle(f'{event} PSTH')

    cbar = fig.colorbar(im, ax=ax)
    vmin, vmax = im.get_clim()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=2, fontsize=8)

    plt.tight_layout()

    if save_psth:
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f'{filename}.png'))

    return


def split_psth(psth, event_idx, event='reward', zscoring=True, time_around=1, funcimg_frame_rate=45):

    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = psth.shape[2]
    num_neurons = psth.shape[0]
    num_events = len(event_idx)

    # Split trials in half (randomly) to confirm event tuning
    num_sort_trials = np.floor(num_events/2).astype(int)
    event_array = np.arange(0, num_events)

    random_rew_sort = np.random.choice(event_array, num_sort_trials, replace=False)  # used for sorting
    random_rew_test = np.setdiff1d(event_array, random_rew_sort)  # used for testing

    # Average firing rates for sort trials and test trials separately
    sorting_data = np.mean(psth[:, random_rew_sort, :], axis=1)
    testing_data = np.mean(psth[:, random_rew_test, :], axis=1)

    if zscoring:
        sorting_data = stats.zscore(sorting_data, axis=1)
        testing_data = stats.zscore(testing_data, axis=1)
        # sorting_data = stats.zscore(sorting_data, axis=None)
        # testing_data = stats.zscore(testing_data, axis=None)
    
    vmin = min(np.min(sorting_data), np.min(testing_data))
    vmax = max(np.max(sorting_data), np.max(testing_data))

    sortidx = np.argsort(np.argmax(sorting_data[:, :], axis=1))

    # Plotting 
    fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])  # third slot for colorbar

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    cax = fig.add_subplot(gs[2])

    im0 = ax0.imshow(sorting_data[sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
    ax0.vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
    ax0.set_xlabel('Time')
    ax0.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
    if time_around == int(time_around):
        xticklabels = [int(-time_around), 0, int(time_around)]
    else:
        xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
    ax0.set_xticklabels(xticklabels)
    ax0.set_title(f'Sorting trials')

    im1 = ax1.imshow(testing_data[sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
    ax1.vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
    ax1.set_xlabel('Time')
    ax1.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
    if time_around == int(time_around):
        xticklabels = [int(-time_around), 0, int(time_around)]
    else:
        xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
    ax1.set_xticklabels(xticklabels)    
    ax1.set_title(f'Testing trials')

    ax0.set_ylabel('Neuron')
    ax0.set_yticks([-0.5, num_neurons-0.5])
    ax0.set_yticklabels([0, num_neurons])

    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_ticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=2, fontsize=8)

    fig.suptitle(f'{event} PSTH')
    plt.tight_layout()


def get_tuned_neurons(psth, event='reward', time_around=1, funcimg_frame_rate=45, plot_neurons=True):
    # Statistics to find neurons tuned to an event e.g. reward, lick, landmark entry etc.
    # TODO: bootstrapping / permutation test instead? 

    # Mann–Whitney U test comparing the period just before stimulus onset to the period directly after stimulus onset. 

    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = psth.shape[2]
    num_neurons = psth.shape[0]

    # Average across timebins for each trial
    before_event_firing = np.mean(psth[:, :, 0:time_window], axis=2)
    after_event_firing = np.mean(psth[:, :, time_window:], axis=2)
    # print(before_event_firing.shape, after_event_firing.shape)

    # Perform the test using all trials for each neuron
    wilcoxon_stat = np.zeros((num_neurons, 1))
    wilcoxon_pval = np.zeros((num_neurons, 1))
    for n in range(num_neurons):
        wilcoxon_stat[n], wilcoxon_pval[n] = stats.wilcoxon(before_event_firing[n, :], after_event_firing[n, :]) #, method=stats.PermutationMethod(n_resamples=1000))

    # Criteria to define tuned neurons
    # 1. p-value
    criterion1 = np.where(wilcoxon_pval < 0.05)[0]   

    # 2. peak in the 1s after event > mean + 2*std of the 1s before the event
    average_psth = np.mean(psth, axis=1)
    before_event_avg_firing = average_psth[:, 0:time_window]
    after_event_avg_firing = average_psth[:, time_window:]
    criterion2 = np.where(np.max(after_event_avg_firing, axis=1) > (np.mean(before_event_avg_firing, axis=1) + 2 * np.std(before_event_avg_firing, axis=1)))[0]

    tuned_neurons = np.intersect1d(criterion1, criterion2)
    print(f'{len(tuned_neurons)} neurons are tuned to {event}.')

    # Plot firing for a few significant neurons
    if plot_neurons:
        for n in tuned_neurons[0:10]:
            fig, ax = plt.subplots(1, 1, figsize=(2,2), sharey=True)
            ax.plot(average_psth[n, :])      
            ax.axvspan(num_timebins/2, num_timebins, color='gray', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            if time_around == int(time_around):
                xticklabels = [int(-time_around), 0, int(time_around)]
            else:
                xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
            ax.set_xticklabels(xticklabels)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylabel('DF/F')

    return tuned_neurons, wilcoxon_stat, wilcoxon_pval


def plot_psth_single_neurons(psth, average_psth, neurons, time_around=(-1, 1), num_neurons=10, avg_only=False, zscoring=True, event_lick_rate=None, pvalues=None, color=None, axis=None):
    '''Plot the PSTH around events for specific neurons.'''

    num_timebins = average_psth.shape[1]
    num_events = psth.shape[1]

    if isinstance(neurons, int) or np.isscalar(neurons):
        neurons = [neurons]

    # Handle time window input
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a single number or a tuple/list of (start, end)")

    # z-scoring
    if zscoring:
        average_psth = stats.zscore(np.array(average_psth), axis=1)       
        psth = stats.zscore(np.array(psth), axis=2)

    for i, n in enumerate(neurons[:num_neurons]):
        if axis is None:
            fig, ax = plt.subplots(1, 1, figsize=(2,2), sharey=True)
        else: 
            ax = axis
        if not avg_only:
            for r in range(num_events):
                ax.plot(psth[n, r, :])
            linewidth = 3
        else:
            linewidth = 2

        if color == 'blue':
            label = '2/3 Alternation'
        elif color == 'red':
            label = '3/3 Discrimination'
        else:
            label = None

        # Plot PSTH
        ax.plot(average_psth[n, :], color=color if color is not None else 'black', linewidth=linewidth, label=label) 

        if avg_only and not zscoring:  # add SEM
            ax.fill_between(np.arange(num_timebins),
                            average_psth[n, :] - stats.sem(psth[n, :, :], axis=0),
                            average_psth[n, :] + stats.sem(psth[n, :, :], axis=0),
                            color=color if color is not None else 'black',
                            alpha=0.3)
        
        # Plot lick rate if available
        if event_lick_rate is not None:
            # Get mean lick rate across events 
            avg_event_lick_rate = np.mean(event_lick_rate, axis=0)
            sem_event_lick_rate = stats.sem(event_lick_rate, axis=0)

            ax2 = ax.twinx()
            ax2.plot(avg_event_lick_rate, color='orange', linestyle='-', label='Lick Rate', linewidth=2)

            ax2.fill_between(np.arange(num_timebins),
                            avg_event_lick_rate - sem_event_lick_rate,
                            avg_event_lick_rate + sem_event_lick_rate,
                            color='orange', alpha=0.3)

            # Label the second y-axis
            ax2.set_ylabel('Lick Rate (Hz)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

        # Event alignment marker (time 0 is at -start_time in bins)
        zero_bin = int(round(-start_time / (end_time - start_time) * num_timebins))
        ax.axvspan(zero_bin, num_timebins, color='gray', alpha=0.5)

        # X-axis labels
        ax.set_xlabel('Time (s)')
        ax.set_xticks([0, zero_bin, num_timebins - 1])
        ax.set_xticklabels([round(start_time, 2), 0, round(end_time, 2)])

        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylabel(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0')

        if pvalues is not None:
            ax.set_title(f'p-value {round(pvalues[i], 3 - int(floor(log10(abs(pvalues[i])))) - 1)}')

    return


def get_tuned_neurons_shohei(DF_F, average_psth, neurons, event='reward', time_around=1, funcimg_frame_rate=45, plot_neurons=True, zscoring=True):
    # The response to an event is calculated using the mean z-scored ΔF/F calcium signal 
    # averaged over a window from 0.4 s to 1 s after event onset, baseline-subtracted using 
    # the mean z-scored ΔF/F signal during 0.5 s before event onset for each event. 
    # Neurons are classified as event-responsive if their mean response is bigger than 0.5 z-scored ΔF/F. 
    
    time_window = time_around * funcimg_frame_rate # frames
    time_before = int(np.floor(0.5 * funcimg_frame_rate))
    time_after = int(0.4 * funcimg_frame_rate)
    num_timebins = average_psth.shape[1]
    num_neurons = average_psth.shape[0]

    num_neurons = len(neurons)

    data = average_psth.copy()
    if zscoring:
        data = stats.zscore(np.array(data), axis=1)
        # data = stats.zscore(np.array(data), axis=None)

    before_firing = data[:, time_before:time_window]
    after_firing = data[:, time_window+time_after:]
    
    mean_before = np.mean(before_firing, axis=1)
    mean_after = np.mean(after_firing, axis=1)

    total_response = mean_after - mean_before

    tuned_neurons = []
    for n in range(num_neurons):
        if total_response[n] > 0.5 * np.mean(DF_F[n,:]):
            tuned_neurons.append(n)
    
    print(f'{len(tuned_neurons)} neurons are tuned to {event}.')

    if plot_neurons:
        # Plot firing for a few significant neurons
        for n in tuned_neurons[0:10]:
            fig, ax = plt.subplots(1, 1, figsize=(2,2), sharey=True)
            ax.plot(average_psth[n, :])      
            ax.axvspan(num_timebins/2, num_timebins, color='gray', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            ax.set_xticklabels([int(-time_around), 0, int(time_around)])
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylabel('DF/F')

    return tuned_neurons


def plot_avg_goal_psth(neurons, event_idxs, psths, average_psths, \
                        goals=['A','B','C','D'], time_around=1, funcimg_frame_rate=45, \
                        plot_all_neurons=False, save_plot=False, savepath='', savedir=''):
    
    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    num_goals = len(goals)

    if plot_all_neurons:
        for n, neuron in enumerate(neurons):

            fig, ax = plt.subplots(1, num_goals, figsize=(10,2), sharey=True, sharex=True)
            ax = ax.ravel()
            
            for goal in range(num_goals):
                psth = psths[goal]
                avg_psth = average_psths[goal]
                event_idx = event_idxs[goal]

                for i in range(len(event_idx)):
                    ax[goal].plot(psth[n, i, :], alpha=0.5)

                ax[goal].plot(avg_psth[n, :], 'k', linewidth=2)
                ax[goal].axvspan(num_timebins / 2, num_timebins, color='gray', alpha=0.5)
                ax[goal].set_xticks([-0.5, num_timebins/2 - 0.5, num_timebins - 0.5])
                ax[goal].set_xticklabels([int(-time_around), 0, int(time_around)])
                ax[goal].spines[['right', 'top']].set_visible(False)
                ax[goal].set_title(goals[goal])

            ax[0].set_ylabel('DF/F')
            plt.suptitle(f'Neuron {neuron}')

            if save_plot:
                output_path = os.path.join(savepath, savedir)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                plt.savefig(os.path.join(output_path, f'neuron{neuron}.png'))
                plt.close()

    else:
        for n, neuron in enumerate(neurons[0:10]):

            fig, ax = plt.subplots(1, num_goals, figsize=(10,2), sharey=True, sharex=True)
            ax = ax.ravel()
            
            for goal in range(num_goals):
                psth = psths[goal]
                avg_psth = average_psths[goal]
                event_idx = event_idxs[goal]

                for i in range(len(event_idx)):
                    ax[goal].plot(psth[n, i, :], alpha=0.5)

                ax[goal].plot(avg_psth[n, :], 'k', linewidth=2)
                ax[goal].axvspan(num_timebins / 2, num_timebins, color='gray', alpha=0.2)
                ax[goal].set_xticks([-0.5, num_timebins/2 - 0.5, num_timebins - 0.5])
                ax[goal].set_xticklabels([int(-time_around), 0, int(time_around)])
                ax[goal].spines[['right', 'top']].set_visible(False)
                ax[goal].set_title(goals[goal])

            ax[0].set_ylabel('DF/F')
            plt.suptitle(f'Neuron {neuron}')
            plt.show()


def get_landmark_psth(data, neurons, event_idx, num_landmarks=10, time_around=1, funcimg_frame_rate=45):
    '''This function is similar to get_psth, but the average PSTH is calculated for each landmark separately.'''

    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    if isinstance(neurons, int) or np.isscalar(neurons):
        neurons = [neurons]

    num_timebins = 2*time_window
    num_neurons = len(neurons)
    num_events = len(event_idx)

    window_indices = np.add.outer(event_idx, np.arange(-time_window, time_window)).astype(int)  

    # Remove last events if close to session end 
    valid_mask = window_indices[:, -1] < data.shape[1]
    valid_window_indices = window_indices[valid_mask]

    # Preallocate PSTH array
    num_events = valid_window_indices.shape[0]
    psth = np.zeros((num_neurons, num_events, num_timebins))
    for n, neuron in enumerate(neurons):
        psth[n, :, :] = data[neuron, valid_window_indices]

    # Average PSTH for all events per landmark
    average_landmark_psth = np.zeros([num_neurons, num_landmarks, num_timebins])
    for i in range(num_landmarks):
        average_landmark_psth[:, i, :] = np.mean(psth[:, i::num_landmarks, :], axis=1)

    return psth, average_landmark_psth


def get_landmark_id_psth(data, neurons, event_idx, session, num_landmarks=2, time_around=1, funcimg_frame_rate=45):
    '''This function is similar to get_psth, but the average PSTH is calculated for each landmark separately.'''

    assert num_landmarks == 2, 'This function only deals with 2 landmark sequences.'

    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = 2*time_window
    num_neurons = len(neurons)
    num_events = len(event_idx)

    window_indices = np.add.outer(event_idx, np.arange(-time_window, time_window)).astype(int)  

    psth = np.zeros((num_neurons, num_events, num_timebins))
    for n, neuron in enumerate(neurons):
        psth[n, :, :] = data[neuron, window_indices]

    # Average PSTH for all events per landmark
    average_landmark_psth = np.zeros([num_neurons, num_landmarks, num_timebins])
    for i in range(num_landmarks):
        if i == 0:
            average_landmark_psth[:, i, :] = np.mean(psth[:, session['goals_idx'], :], axis=1)
        elif i == 1:
            average_landmark_psth[:, i, :] = np.mean(psth[:, session['non_goals_idx'], :], axis=1)

    return psth, average_landmark_psth


def plot_avg_landmark_psth(neurons, psth, average_psth, num_landmarks=10, time_around=1, funcimg_frame_rate=45, \
                           plot_all_neurons=False, save_plot=False, savepath='', savedir=''):
    
    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    if plot_all_neurons:
        for n, neuron in enumerate(neurons):

            fig, ax = plt.subplots(1, 10, figsize=(15, 2), sharey=True, sharex=True)
            ax = ax.ravel()

            for i in range(num_landmarks):
                ax[i].plot(psth[n, i::num_landmarks, :].T, alpha=0.5)  
                ax[i].plot(average_psth[n, i, :], 'k', linewidth=3)
                ax[i].axvspan(num_timebins/2, num_timebins, color='gray', alpha=0.5)
                ax[i].set_xlabel('Time')
                ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
                ax[i].set_xticklabels([int(-time_around), 0, int(time_around)])
                ax[i].spines[['right', 'top']].set_visible(False)

            ax[0].set_ylabel('DF/F')
            plt.tight_layout()
            plt.suptitle(f'Neuron {neuron}')
        
            if save_plot:
                output_path = os.path.join(savepath, savedir)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                plt.savefig(os.path.join(output_path, f'neuron{neuron}.png'))
                plt.close()

    else:
        for n, neuron in enumerate(neurons[0:10]):

            fig, ax = plt.subplots(1, 10, figsize=(15, 2), sharey=True, sharex=True)
            ax = ax.ravel()

            for i in range(num_landmarks):
                ax[i].plot(psth[n, i::num_landmarks, :].T)  # TODO: confirm indices
                ax[i].plot(average_psth[n, i, :], 'k', linewidth=3)
                ax[i].axvspan(num_timebins/2, num_timebins, color='gray', alpha=0.5)
                ax[i].set_xlabel('Time')
                ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
                ax[i].set_xticklabels([int(-time_around), 0, int(time_around)])
                ax[i].spines[['right', 'top']].set_visible(False)

            ax[0].set_ylabel('DF/F')
            plt.tight_layout()
            plt.suptitle(f'Neuron {neuron}')


def plot_landmark_psth_map(average_psth, session, zscoring=True, sorting_lm=0, num_landmarks=10, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir='', filename=''):
    '''Plot firing maps of all selected neurons for all landmarks, sorted by specific landmark.'''

    if sorting_lm >= num_landmarks:
        raise ValueError(f'The sorting landmark should be one of the {num_landmarks} landmarks.')
    
    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = average_psth.shape[2]

    fig, ax = plt.subplots(1, num_landmarks, figsize=(num_landmarks*1.5+2,3), sharey=True, sharex=True)
    ax = ax.ravel()

    data = average_psth.copy()
    if zscoring:
        data = stats.zscore(data, axis=1)
        # data = stats.zscore(data, axis=None)

    vmin = min([np.nanmin(data)])
    vmax = max([np.nanmax(data)])

    sortidx = np.argsort(np.argmax(data[:, sorting_lm, :], axis=1))

    for i in range(num_landmarks):
        img = ax[i].imshow(data[sortidx, i, :], aspect='auto', vmin=vmin, vmax=vmax)
        ax[i].vlines(time_window-0.5, ymin=-0.5, ymax=data.shape[0]-0.5, color='k', linewidth=0.5)
        ax[i].set_xlabel('Time')
        ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
        if time_around == int(time_around):
            xticklabels = [int(-time_around), 0, int(time_around)]
        else:
            xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
        ax[i].set_xticklabels(xticklabels)
        ax[i].spines[['right', 'top']].set_visible(False)
        if num_landmarks == 10:
            ax[i].set_title(f'{i+1}')
        else:
            lm = session['all_lms'][session['goals_idx'][0]] if i == 0 else session['all_lms'][session['non_goals_idx'][0]] 
            ax[i].set_title(f'{lm+1}')

    ax[0].set_yticks([-0.5, data.shape[0]-0.5])
    ax[0].set_yticklabels([0, data.shape[0]])
    ax[0].set_ylabel('Neuron', labelpad=-10)

    cbar = fig.colorbar(img, ax=ax.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=10, fontsize=8)

    # plt.tight_layout()

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'{filename}.png'))
        plt.show()


def plot_goal_psth_map(average_psths, zscoring=True, sorting_goal=1, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir='', filename=''):
    '''Plot firing maps of all selected neurons for each goal, sorted by specific goal.'''

    num_goals = len(average_psths)
    if num_goals == 4:
        goals = ['A','B','C','D']
    else:
        goals = ['A','B']

    if sorting_goal not in average_psths:
        raise ValueError(f'The sorting landmark should be one of the {num_goals} landmarks.')
    
    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    data = average_psths.copy()
    if zscoring:
        for goal in data.keys():
            data[goal] = stats.zscore(data[goal], axis=1)
            # data[goal] = stats.zscore(data[goal], axis=None)

    # Find global vmin and vmax across all goals
    vmin = min([np.nanmin(arr) for arr in data.values()])
    vmax = max([np.nanmax(arr) for arr in data.values()])

    im = [[] for _ in range(num_goals)]
    fig, ax = plt.subplots(1, num_goals, figsize=(3*num_goals, 4), sharey=True, sharex=True)
    ax = ax.ravel()

    sortidx = np.argsort(np.argmax(data[sorting_goal], axis=1))  # expects a dict with keys = goals

    for i, goal in enumerate(sorted(data.keys())):
        im[i] = ax[i].imshow(data[goal][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
        ax[i].vlines(time_window-0.5, ymin=-0.5, ymax=data[goal].shape[0]-0.5, color='k', linewidth=0.5)
        ax[i].set_xlabel('Time')
        ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
        ax[i].set_xticklabels([int(-time_around), 0, int(time_around)])
        ax[i].spines[['right', 'top']].set_visible(False)
        ax[i].set_title(goals[i])

    ax[0].set_yticks([-0.5, data[goal].shape[0]-0.5])
    ax[0].set_yticklabels([0, data[goal].shape[0]])
    ax[0].set_ylabel('Neuron')

    cbar = fig.colorbar(im[-1], ax=fig.axes, shrink=0.6)

    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=10, fontsize=8)
    
    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'{filename}.png'))
        plt.show()


def plot_all_sessions_goal_psth_map(all_average_psths, conditions, zscoring=True, ref_session=0, sorting_goal=1, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir='', filename=''):
    '''Plot firing maps for all sessions and each goal, sorted by a specific goal. 
    If there is one goal per session, or the average across goals, it behaves like plot_condition_psth_map.'''

    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    num_sessions = len(all_average_psths)

    # Copy and optionally z-score data
    data = []
    goals_per_session = [[] for _ in range(num_sessions)]
    if isinstance(all_average_psths, list):
        for s, session in enumerate(all_average_psths):
            if isinstance(session, dict):
                session_data = {}
                for goal in session.keys():
                    session_data[goal] = stats.zscore(session[goal], axis=1) if zscoring else session[goal]
                    # session_data[goal] = stats.zscore(session[goal], axis=None) if zscoring else session[goal]
                data.append(session_data)
            else:  # transform data to follow the same structure
                session_data = {}
                session_data[1] = stats.zscore(session, axis=1) if zscoring else session
                # session_data[1] = stats.zscore(session, axis=None) if zscoring else session
                data.append(session_data)
            
            goals_per_session[s] = list(session_data.keys())

    elif isinstance(all_average_psths, dict):
        # Flatten the data
        for session_id, session in all_average_psths.items():  
            if isinstance(session, dict):
                assert sorting_goal in all_average_psths[ref_session], 'This goal does not exist in the reference session.'

                session_data = {}
                for goal in session.keys():
                    session_data[goal] = stats.zscore(session[goal], axis=1) if zscoring else session[goal]
                    # session_data[goal] = stats.zscore(session[goal], axis=None) if zscoring else session[goal]
                data.append(session_data)

            else:  # transform data to follow the same structure
                session_data = {}
                session_data[1] = stats.zscore(session, axis=1) if zscoring else session
                # session_data[1] = stats.zscore(session, axis=None) if zscoring else session
                data.append(session_data)

        goals_per_session = [sorted(data[s].keys()) for s in range(num_sessions)]

    # Compute global vmin/vmax
    vmin = min([np.nanmin(session[goal]) for session in data for goal in session.keys()])
    vmax = max([np.nanmax(session[goal]) for session in data for goal in session.keys()])

    # Sort neurons consistently across sessions (using sorting_goal)
    sortidx = np.argsort(np.argmax(data[ref_session][sorting_goal], axis=1))  # reference the first session for sorting

    # === Plotting ===
    # Set up figure
    max_goals = max(len(goals) for goals in goals_per_session)
    goal_label_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', '1': 'A', '2': 'B', '3': 'C', '4': 'D', 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}
    protocol_nums = sorted(set([cond.split()[0] for cond in conditions]))
    ylabel = [f'{protocol_nums[s]}\nNeuron' if max_goals > 1 else 'Neuron' for s in range(num_sessions)]
    # titles = [protocol_nums[s] for s in range(num_sessions)]
    titles = [conditions[s] for s in range(num_sessions)]

    if max_goals == 1: 
        nrows = 1
        ncols = num_sessions
    else:
        nrows = num_sessions
        ncols = max_goals
        
    fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows), sharex=True, sharey=True)

    # if num_sessions == 1 or max_goals == 1:
    ax = np.atleast_2d(ax)
    ax = np.array(ax)

    for s in range(num_sessions):
        for g, goal in enumerate(goals_per_session[s]):
            if max_goals == 1:  # one row, multiple columns
                row = 0
                col = s  
            else:
                row = s
                col = g  
            ax[row, col].imshow(data[s][goal][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
            ax[row, col].vlines(time_window-0.5, ymin=-0.5, ymax=data[s][goal].shape[0]-0.5, color='k', linewidth=0.5)
            ax[row, col].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            if time_around == int(time_around):
                xticklabels = [int(-time_around), 0, int(time_around)]
            else:
                xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
            ax[row, col].set_xticklabels(xticklabels)
            ax[row, col].spines[['right', 'top']].set_visible(False)
            if max_goals != 1:
                ax[row, col].set_title(goal_label_map.get(goal, str(goal)))
            else:
                ax[row, col].set_title(titles[s])
            
        ax[row,0].set_ylabel(ylabel[s], labelpad=-5)
        ax[row,0].set_yticks([-0.5, data[ref_session][goals_per_session[0][0]].shape[0]-0.5])  
        ax[row,0].set_yticklabels([0, data[ref_session][goals_per_session[0][0]].shape[0]])
            
        # Hide unused axes in that row
        for g_unused in range(len(goals_per_session[s]), max_goals):
            ax[s, g_unused].axis('off')

    cbar = fig.colorbar(ax[0,0].images[0], ax=ax.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=0, fontsize=8)

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'{filename}.png'))
    plt.show()


def plot_condition_psth_map(average_psths, conditions, zscoring=True, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir=''):
    '''Compare average PSTH map across different conditions.'''

    time_window = time_around * funcimg_frame_rate # frames
    # num_timebins = 2*time_window
    num_timebins = average_psths[0].shape[1]
    num_neurons = average_psths[0].shape[0]

    data = [[] for i in range(len(conditions))]
    for i in range(len(conditions)):
        data[i] = average_psths[i].copy()
        if zscoring:
            data[i] = stats.zscore(data[i], axis=1)
            # data[i] = stats.zscore(data[i], axis=None)

    # Find global vmin and vmax across all conditions
    vmin = min([np.nanmin(d) for d in data if d.size > 0])
    vmax = max([np.nanmax(d) for d in data if d.size > 0])

    # === Plotting ===
    for c, condition in enumerate(conditions):
        sortidx = np.argsort(np.argmax(data[c], axis=1))  # Sort by different conditions
        
        im = [[] for _ in range(len(conditions))]
        fig, ax = plt.subplots(1, len(conditions), figsize=(3*len(conditions),3), sharex=True, sharey=True)
        ax = ax.ravel()
        
        for i in range(len(conditions)):
            im[i] = ax[i].imshow(data[i][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)    
            ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            if time_around == int(time_around):
                xticklabels = [int(-time_around), 0, int(time_around)]
            else:
                xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
            ax[i].set_xticklabels(xticklabels, fontsize=8)
            ax[i].spines[['right', 'top']].set_visible(False)
            ax[i].set_xlabel('Time', fontsize=8)
            ax[i].set_title(f'{conditions[i]}', fontsize=10)
            ax[i].vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
        
        ax[0].set_yticks([-0.5, num_neurons-0.5])
        ax[0].set_yticklabels([0, num_neurons], fontsize=8)
        ax[0].set_ylabel('Neuron', fontsize=8, labelpad=-5)

        cbar = fig.colorbar(im[-1], ax=ax.ravel().tolist(), shrink=0.6)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
        cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=10, fontsize=8)

        plt.suptitle(f'Sorting by {condition} trials', fontsize=10)

        if save_plot:
            output_path = os.path.join(savepath, savedir)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, f'{condition}_sorting.png'))
        plt.show()
        

def get_rolling_map_correlation(average_psths, conditions, population=False, zscoring=True, color_scheme=None, ax=None, save_plot=False, savepath='', savedir='', filename=''):
    '''
    Get the firing map correlation among different conditions against a reference. 
    The correlation for the reference is calculated by randomly selecting half the trials.
    If population is True, the correlation is computed across the entire activity map. Otherwise it is calculated on a neuron-by-neuron basis.
    NOTE: The reference is the index of the data if the data are either a list or a nested dict (will get flattened into a list), but it is a key of the data if the data are a dict. 
    '''
    num_neurons = average_psths[0].shape[0]
    num_windows = average_psths[0].shape[1]
    num_timebins = average_psths[0].shape[2]
    num_conditions = len(conditions)

    if zscoring:
        average_psths = [stats.zscore(psth, axis=2) for psth in average_psths]
    
    # 1. Within-condition correlations (rolling across windows)
    within_corrs = [[[] for _ in range(num_windows - 1)] for _ in range(num_conditions)]

    # 2. Across-condition correlations (between same windows)
    condition_pairs = list(itertools.combinations(range(num_conditions), 2))
    across_corrs = [[[] for _ in range(num_windows)] for _ in range(len(condition_pairs))]

    # Calculate within-condition rolling correlations
    for c in range(num_conditions):
        for i in range(num_windows - 1):
            if population:
                for t in range(num_timebins):
                    v1 = average_psths[c][:, i, t]
                    v2 = average_psths[c][:, i + 1, t]
                    if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                        r, _ = stats.pearsonr(v1, v2)
                        within_corrs[c][i].append(r)
                    else:
                        within_corrs[c][i].append(np.nan)
            else:
                for n in range(num_neurons):
                    v1 = average_psths[c][n, i, :]
                    v2 = average_psths[c][n, i + 1, :]
                    if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                        r, _ = stats.pearsonr(v1, v2)
                        within_corrs[c][i].append(r)
                    else:
                        within_corrs[c][i].append(np.nan)

    # Calculate across-condition correlations (same window index)
    for pair_idx, (c1, c2) in enumerate(condition_pairs):
        for i in range(num_windows):
            if population:
                for t in range(num_timebins):
                    v1 = average_psths[c1][:, i, t]
                    v2 = average_psths[c2][:, i, t]
                    if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                        r, _ = stats.pearsonr(v1, v2)
                        across_corrs[pair_idx][i].append(r)
                    else:
                        across_corrs[pair_idx][i].append(np.nan)
            else:
                for n in range(num_neurons):
                    v1 = average_psths[c1][n, i, :]
                    v2 = average_psths[c2][n, i, :]
                    if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                        r, _ = stats.pearsonr(v1, v2)
                        across_corrs[pair_idx][i].append(r)
                    else:
                        across_corrs[pair_idx][i].append(np.nan)

    # === Plotting ===
    # 1. Each neuron correlation trace & mean
    fig, ax = plt.subplots(1, len(within_corrs), figsize=(8,3))
    ax = ax.ravel()

    for i in range(len(within_corrs)):
        mean_across_neurons = [np.mean(within_corrs[i][w]) for w in range(len(within_corrs[i]))]
        ax[i].plot(within_corrs[i])
        ax[i].plot(mean_across_neurons, color='black')
        ax[i].set_title(f"{conditions[i]} vs {conditions[i]}")

    fig, ax = plt.subplots(1, len(across_corrs), figsize=(4,3))
    if len(across_corrs) > 1:
        ax = ax.ravel()
    for i in range(len(across_corrs)):
        mean_across_neurons = [np.mean(across_corrs[i][w]) for w in range(len(across_corrs[i]))]
        if len(across_corrs) > 1:
            ax.plot(across_corrs[i])
            ax.plot(mean_across_neurons, color='black')
            ax.set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
        else:
            ax.plot(across_corrs[i])
            ax.plot(mean_across_neurons, color='black')
            ax.set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")

    # 2. Mean +/- sem correlation trace 
    fig, ax = plt.subplots(1, len(within_corrs) + len(across_corrs), figsize=(12,3), sharey=True, sharex=True)
    ax = ax.ravel()

    k = 0 
    for i in range(len(within_corrs)):
        mean_across_neurons = np.array([np.mean(within_corrs[i][w]) for w in range(len(within_corrs[i]))])
        sem_across_neurons = stats.sem(np.array([within_corrs[i][w] for w in range(len(within_corrs[i]))]), axis=1)

        ax[k].fill_between(np.arange(len(within_corrs[i])),
                        mean_across_neurons - sem_across_neurons,
                        mean_across_neurons + sem_across_neurons,
                        color='black',
                        alpha=0.3)
        # ax[i].plot(within_corrs[i])
        ax[k].plot(mean_across_neurons, color='black')
        ax[k].set_title(f"{conditions[i]} vs {conditions[i]}")
        ax[k].set_xlabel('Lap block')
        k += 1

    for j in range(len(across_corrs)):
        mean_across_neurons = np.array([np.mean(across_corrs[j][w]) for w in range(len(across_corrs[j]))])
        sem_across_neurons = stats.sem(np.array([across_corrs[j][w] for w in range(len(across_corrs[j]))]), axis=1)

        ax[k].plot(mean_across_neurons, color='black')
        ax[k].fill_between(np.arange(len(across_corrs[j])),
                    mean_across_neurons - sem_across_neurons,
                    mean_across_neurons + sem_across_neurons,
                    color='black',
                    alpha=0.3)
        ax[k].set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
        ax[k].set_xlabel('Lap block')
        k += 1

    # TODO: add saving options
        
    return within_corrs, across_corrs, condition_pairs


def get_window_similarity_matrix(average_psths, conditions, population=False, zscoring=True, plot=True):
    """
    Compute full window-by-window correlation matrices for each condition.
    Returns one similarity matrix per condition.

    Parameters:
        average_psths: list of np.arrays (num_neurons x num_windows x num_timebins) per condition
        conditions: list of condition names (same order as average_psths)
        population: if True, correlate population vectors, otherwise per-neuron average
        zscoring: whether to z-score each neuron's time series before computing similarity
        plot: whether to show the matrices

    Returns:
        similarity_matrices: list of np.arrays (num_windows x num_windows) for each condition
    """
    num_neurons = average_psths[0].shape[0]
    num_windows = average_psths[0].shape[1]
    num_timebins = average_psths[0].shape[2]
    num_conditions = len(conditions)

    if zscoring:
        average_psths = [stats.zscore(psth, axis=2) for psth in average_psths]

    similarity_matrices = []

    # Compute within condition similarity matrix
    for c, psth in enumerate(average_psths):

        if population:
            # Store one sim_matrix per timebin, then average
            sim_matrix_all_timebins = np.full((num_timebins, num_windows, num_windows), np.nan)

            for i in range(num_windows):
                for j in range(num_windows):
                    for t in range(num_timebins):
                        v1 = psth[:, i, t]
                        v2 = psth[:, j, t]
                        if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                            r, _ = stats.pearsonr(v1, v2)
                            sim_matrix_all_timebins[t, i, j] = r
                        else:
                            sim_matrix_all_timebins[t, i, j] = np.nan
                    
            # Average across timebins
            sim_matrix = np.nanmean(sim_matrix_all_timebins, axis=0)

        else:
            # Store one sim_matrix per neuron, then average
            sim_matrix_all_neurons = np.full((num_neurons, num_windows, num_windows), np.nan)

            for i in range(num_windows):
                for j in range(num_windows):
                    for n in range(num_neurons):
                        v1 = psth[n, i, :]
                        v2 = psth[n, j, :]
                        if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                            r, _ = stats.pearsonr(v1, v2)
                            sim_matrix_all_neurons[n, i, j] = r
                        else:
                            sim_matrix_all_neurons[n, i, j] = np.nan

            # Average across neurons
            sim_matrix = np.nanmean(sim_matrix_all_neurons, axis=0)

        similarity_matrices.append(sim_matrix)

    # Set diagonal to nan to avoid skewing the colormap
    for sim_matrix in similarity_matrices:
        np.fill_diagonal(sim_matrix, np.nan)
        
    # Compute across condition similarity matrix
    condition_pairs = list(itertools.combinations(range(num_conditions), 2))
    for pair_idx, (c1, c2) in enumerate(condition_pairs):

        if population:
            # Store one sim_matrix per timebin, then average
            sim_matrix_all_timebins = np.full((num_timebins, num_windows, num_windows), np.nan)

            for i in range(num_windows):
                for j in range(num_windows):                    
                    for t in range(num_timebins):
                        v1 = average_psths[c1][:, i, t]
                        v2 = average_psths[c2][:, j, t]
                        if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                            r, _ = stats.pearsonr(v1, v2)
                            sim_matrix_all_timebins[t, i, j] = r
                        else:
                            sim_matrix_all_timebins[t, i, j] = np.nan
                    
            # Average across timebins
            sim_matrix = np.nanmean(sim_matrix_all_timebins, axis=0)

        else:
            # Store one sim_matrix per neuron, then average
            sim_matrix_all_neurons = np.full((num_neurons, num_windows, num_windows), np.nan)

            for i in range(num_windows):
                for j in range(num_windows): 
                    for n in range(num_neurons):
                        v1 = average_psths[c1][n, i, :]
                        v2 = average_psths[c2][n, j, :]
                        if np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                            r, _ = stats.pearsonr(v1, v2)
                            sim_matrix_all_neurons[n, i, j] = r
                        else:
                            sim_matrix_all_neurons[n, i, j] = np.nan

            # Average across neurons
            sim_matrix = np.nanmean(sim_matrix_all_neurons, axis=0)

        similarity_matrices.append(sim_matrix)

        # Optional plot
        if plot:
            # Individual colormaps
            fig, ax = plt.subplots(1, len(similarity_matrices), figsize=(12,4), sharex=True, sharey=True)
            ax = ax.ravel()
            
            for i, sim_matrix in enumerate(similarity_matrices):
                vmax = np.round(np.nanmax(sim_matrix), 2)
                if vmax < 1e-3:
                    vmax = 1e-3
                vmin = -vmax
                im = ax[i].imshow(sim_matrix, vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
                
                cb = fig.colorbar(im, ax=ax[i], shrink=0.8, ticks=[vmin, vmax])  
                cb.set_label('Correlation (r)', labelpad=-5)

                if i < len(average_psths):
                    ax[i].set_title(f"{conditions[i]} vs {conditions[i]}")
                else:
                    ax[i].set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
                ax[i].set_xlabel("Lap block")
                ax[i].set_ylabel("Lap block")
            plt.tight_layout()

            # Global colormap
            all_values = np.concatenate([sim[~np.isnan(sim)].flatten() for sim in similarity_matrices])
            vmax = np.round(np.max(all_values), 2)
            vmin = -vmax
            
            fig, ax = plt.subplots(1, len(similarity_matrices), figsize=(12,4), sharex=True, sharey=True)
            ax = ax.ravel()
            
            for i, sim_matrix in enumerate(similarity_matrices):
                im = ax[i].imshow(sim_matrix, vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')

                if i < len(average_psths):
                    ax[i].set_title(f"{conditions[i]} vs {conditions[i]}")
                else:
                    ax[i].set_title(f"{conditions[int(condition_pairs[0][0])]} vs {conditions[int(condition_pairs[0][1])]}")
                ax[i].set_xlabel("Lap block")
                ax[i].set_ylabel("Lap block")
            # plt.tight_layout()
            fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, label='Correlation (r)', ticks=[vmin, vmax])

    return similarity_matrices


def get_map_correlation(psths, average_psths, conditions, population=False, zscoring=True, reference=0, color_scheme=None, ax=None, save_plot=False, savepath='', savedir='', filename=''):
    '''
    Get the firing map correlation among different conditions against a reference. 
    The correlation for the reference is calculated by randomly selecting half the trials.
    If population is True, the correlation is computed across the entire activity map. Otherwise it is calculated on a neuron-by-neuron basis.
    NOTE: The reference is the index of the data if the data are either a list or a nested dict (will get flattened into a list), but it is a key of the data if the data are a dict. 
    '''
    # Check data format
    if isinstance(average_psths, list):
        if reference > len(conditions):
            raise ValueError('The reference data should be within the range of input average PSTHs.')
    
        average_psth_data = []
        psth_data = []
        # psth_data = [psths[c] for c in range(len(conditions))]
        if zscoring:
            # average_psth_data = stats.zscore(np.array(average_psth_data), axis=2)
            for c in range(len(conditions)):
                average_psth_data.append(stats.zscore(np.array(average_psths[c]), axis=1))
                # average_psth_data.append(stats.zscore(np.array(average_psths[c]), axis=None))
                # psth_data = stats.zscore(np.array(psth_data), axis=2)
                psth_data.append(stats.zscore(np.array(psths[c]), axis=2))
                # psth_data.append(stats.zscore(np.array(psths[c]), axis=None))
        else: 
            average_psth_data = [average_psths[c] for c in range(len(conditions))]
            psth_data = [psths[c] for c in range(len(conditions))]

        data_indices = np.arange(0, len(conditions))
        ref_cond = reference

    elif isinstance(average_psths, dict):
        first_entry = next(iter(average_psths))  

        if isinstance(average_psths[first_entry], dict):
            # Flatten all data: [(session 0 goal A), (session 0 goal B), ..., (session 1 goal A), ...]
            average_psth_data = []  
            psth_data = []  
            for s in average_psths.keys():
                for goal in average_psths[s].keys():  
                    d = average_psths[s][goal]
                    ref = psths[s][goal]
                    if zscoring:
                        d = stats.zscore(d, axis=1)  
                        # d = stats.zscore(d, axis=None)  
                        ref = stats.zscore(ref, axis=2)
                        # ref = stats.zscore(ref, axis=None)
                    average_psth_data.append(d)
                    psth_data.append(ref)

            assert len(average_psth_data) == len(conditions), 'The length of the input data does not match the number of conditions.'
            
            # Create array of indexing into the data 
            data_indices = np.arange(0, len(average_psth_data))
            if reference not in data_indices:
                raise ValueError(f'Reference condition {reference} should be within the range of input average PSTHs.')
            ref_cond = reference
            
        else:
            average_psth_data = average_psths.copy()
            psth_data = psths.copy()
            if zscoring:
                for i in average_psth_data.keys():  
                    average_psth_data[i] = stats.zscore(average_psth_data[i], axis=1)
                    # average_psth_data[i] = stats.zscore(average_psth_data[i], axis=None)
                    psth_data[i] = stats.zscore(psth_data[i], axis=2)
                    # psth_data[i] = stats.zscore(psth_data[i], axis=None)

            data_indices = list(average_psth_data.keys())
            if reference not in average_psth_data.keys():
                raise ValueError(f'Reference condition {reference} should be one of the keys of the input dict.')
            ref_cond = data_indices.index(reference)

    num_neurons = average_psth_data[reference].shape[0]
    num_timebins = average_psth_data[reference].shape[1]
    
    corrs = [[] for c in data_indices]

    # Split reference PSTH data into random half trials 
    num_sort_trials = np.floor(psth_data[reference].shape[1]/2).astype(int)
    event_array = np.arange(0, psth_data[reference].shape[1])

    random_rew_sort = np.random.choice(event_array, num_sort_trials, replace=False)  # used for sorting
    random_rew_test = np.setdiff1d(event_array, random_rew_sort)  # used for testing

    sorting_data = np.mean(psth_data[reference][:, random_rew_sort, :], axis=1)
    testing_data = np.mean(psth_data[reference][:, random_rew_test, :], axis=1)

    # Calculate correlations
    for c, idx in enumerate(data_indices):
        if population is True:
            for t in range(num_timebins):
                if idx == reference:
                    if np.all(np.isfinite(sorting_data[:,t])) and np.all(np.isfinite(testing_data[:,t])):
                        r, _ = stats.pearsonr(sorting_data[:,t], testing_data[:,t])
                        corrs[c].append(r)
                    else:
                        corrs[c].append(np.nan)
                else:
                    if np.all(np.isfinite(average_psth_data[reference][:,t])) and np.all(np.isfinite(average_psth_data[idx][:,t])):
                        r, _ = stats.pearsonr(average_psth_data[reference][:,t], average_psth_data[idx][:,t])
                        corrs[c].append(r)
                    else:
                        corrs[c].append(np.nan)
        else:
            for n in range(num_neurons):
                if idx == reference:
                    if np.all(np.isfinite(sorting_data[n])) and np.all(np.isfinite(testing_data[n])):
                        r, _ = stats.pearsonr(sorting_data[n], testing_data[n])
                        corrs[c].append(r)
                    else:
                        corrs[c].append(np.nan)
                else:
                    if np.all(np.isfinite(average_psth_data[reference][n])) and np.all(np.isfinite(average_psth_data[idx][n])):
                        r, _ = stats.pearsonr(average_psth_data[reference][n], average_psth_data[idx][n])
                        corrs[c].append(r)
                    else:
                        corrs[c].append(np.nan)
    
    # Convert to numpy arrays
    for c in range(len(conditions)):
        corrs[c] = np.array(corrs[c])

    # === Plotting ===
    # Set up labels
    labels = []
    for i, cond in enumerate(conditions):
        if isinstance(average_psths, list):
            labels.append(f"{cond}\nvs\n{conditions[ref_cond]}")
        elif isinstance(average_psths, dict):
            if len(cond) > 10:
                labels.append(f"{cond}\nvs\n{conditions[ref_cond]}")
            else:
                labels.append(f"{cond} vs {conditions[ref_cond]}")

    if color_scheme is None:
        color_scheme = sns.color_palette("Set2", len(corrs))   # Fallback color scheme if none is given

    # Compute mean and SEM for each condition's correlations
    bar_data = []
    sem_data = []
    for c in corrs:
        if np.all(np.isnan(c)):
            bar_data.append(0.0)          
            sem_data.append(0.0)          
        else:
            bar_data.append(np.nanmean(c))
            sem_data.append(stats.sem(c[~np.isnan(c)]) if np.sum(~np.isnan(c)) > 1 else 0)

    # Plot    
    if ax is None: 
        _, ax = plt.subplots(figsize=(len(corrs)+1, 4))
        ax.set_ylabel('Mean correlation')
        if population is True:
            ax.set_title('Population vector correlations')
        else:
            ax.set_title('Per-neuron PSTH correlations')
    ax.bar(labels, bar_data, yerr=sem_data, capsize=3, color=color_scheme)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if population:
            plt.savefig(os.path.join(output_path, filename + '_population.png'))
        else:
            plt.savefig(os.path.join(output_path, filename + '.png'))

    return corrs


def get_map_correlation_matrix(all_average_psths, conditions, population=False, zscoring=True, save_plot=False, savepath='', savedir='', filename=''):
    '''
    Calculate pairwise PSTH correlation across all sessions and goals. 
    If population is True, the correlation is computed across the entire activity map. Otherwise it is calculated on a neuron-by-neuron basis.
    '''
    num_sessions = len(all_average_psths)

    # Flatten all data: [(session 0 goal A), (session 0 goal B), ..., (session 1 goal A), ...]
    data = []
    for s in range(num_sessions):
        for goal in all_average_psths[s].keys():  
            d = all_average_psths[s][goal]
            if zscoring:
                d = stats.zscore(d, axis=1)  # z-score along time
                # d = stats.zscore(d, axis=None)
            data.append(d)

    num_conditions = len(data)  
    assert num_conditions == len(data), 'The length of the input data does not match the number of conditions.'

    # Initialize correlation matrix
    correlation_matrix = np.zeros((num_conditions, num_conditions))

    # Calculate correlations
    for i in range(num_conditions):
        for j in range(num_conditions):
            correlations = []
            if population is True:
                for t in range(data[i].shape[1]):
                    if np.all(np.isfinite(data[i][:,t])) and np.all(np.isfinite(data[j][:,t])):
                        r, _ = stats.pearsonr(data[i][:,t], data[j][:,t])
                        correlations.append(r)
                if correlations:
                    correlation_matrix[i,j] = np.nanmean(correlations)
                else:
                    correlation_matrix[i,j] = np.nan  # If no valid timebins
            else:
                for n in range(data[i].shape[0]):  # loop over neurons
                    if np.all(np.isfinite(data[i][n])) and np.all(np.isfinite(data[j][n])):
                        r, _ = stats.pearsonr(data[i][n], data[j][n])
                        correlations.append(r)
                if correlations:
                    correlation_matrix[i,j] = np.nanmean(correlations)
                else:
                    correlation_matrix[i,j] = np.nan  # If no valid neurons

    # === Plot ===
    fig, ax = plt.subplots(figsize=(5,4))
    if population:
        label = 'Mean population correlation'
    else:
        label = 'Mean neuron correlation'
    im = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='bwr', vmin=-1, vmax=1,
            cbar_kws={'label': label}, cbar=False, square=True, annot_kws={"size": 8}, 
            xticklabels=[f"{c}" for c in conditions],
            yticklabels=[f"{c}" for c in conditions])

    cbar = fig.colorbar(im.collections[0], ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label(label, fontsize=10, rotation=270, labelpad=10)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['-1', '0', '1'])
    ax.set_title('All Sessions and Goals PSTH Correlation')

    plt.tight_layout()

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if population:
            plt.savefig(os.path.join(output_path, filename + '_population.png'))
        else:
            plt.savefig(os.path.join(output_path, filename + '.png'))

    plt.show()

    return correlation_matrix


#%% ########  BEHAVIOUR ########
def load_vr_session_info(sess_data_path, VR_data=None, options=None):  # TODO: deprecated? 
    '''Get landmark, goal, and lap information from VR data.'''

    # Load VR data 
    if VR_data is None and options is None:
        VR_data, options = load_vr_behaviour_data(sess_data_path)

    #### Determine behaviour stage: (1) what defines VR start and (2) number of distinct landmarks
    rulename = options['sequence_task']['rulename']
    if rulename == 'run-auto' or rulename == 'run-lick':  # stages 1-2
        start_odour = False  # VR started with reward delivery
    elif rulename == 'olfactory_shaping' or rulename == 'olfactory_test':  # stages 3-6
        start_odour = True  # first VR event was the odour delivery prep

        if rulename == 'olfactory_test':
            num_landmarks = 10
        else:
            num_landmarks = 2
            # print('Please specify the number of landmarks in the corridor!')  # TODO: read this from config file
    
    #### Deal with VR data from a table with Time, Position, Event, TotalRunDistance
    _, position, _, total_dist = get_position_info(VR_data)
    corrected_position = position - np.array(options['flip_tunnel']['margin_start'])

    goals = np.array(options['flip_tunnel']['goals']) #- np.array(options['flip_tunnel']['margin_start'])
    landmarks = np.array(options['flip_tunnel']['landmarks']) #- np.array(options['flip_tunnel']['margin_start'])
    tunnel_length = options['flip_tunnel']['length']

    num_laps = np.ceil([total_dist.max()/position.max()])
    # num_laps = np.ceil([total_dist.max()/corrected_position.max()])
    num_laps = num_laps.astype(int)[0]
    print(f'{num_laps} laps were completed.')

    # find the last landmark that was run through
    last_landmark = np.where(landmarks[:,0] < position[-1])[0][-1]
    num_lms = len(landmarks)*(num_laps-1) + last_landmark 

    lm_ids =  np.array(options['flip_tunnel']['landmarks_sequence'])
    goal_ids = np.array(options['goal_ids'])
    all_lms = np.array([])
    all_goals = np.array([])
    for i in range(num_laps):
        all_lms = np.append(all_lms, lm_ids)
        all_goals = np.append(all_goals, goal_ids)
    all_lms = all_lms.astype(int)
    all_goals = all_goals.astype(int)
    all_lms = all_lms[:num_lms]
    all_goals = all_goals[:num_lms]

    # create a variable that indexes the laps by finding flips first
    flip_ix = np.where(np.diff(position) < -50)[0]
    # a lap is between two flips
    lap_num = np.zeros(len(position))
    for i in range(len(flip_ix)-1):
        lap_num[flip_ix[i]:flip_ix[i+1]] = i+1
    if num_laps > 1:
        lap_num[flip_ix[-1]:] = len(flip_ix)

    # find the landmarks that were completed
    total_lm_position = np.array([])
    for i in range(num_laps):
        lap_lms = landmarks + i*tunnel_length
        total_lm_position = np.append(total_lm_position, lap_lms[:,0])
    total_lm_position = total_lm_position[:num_lms].astype(int)
    print(f"{total_lm_position.shape[0]} landmarks were visited")

    return num_landmarks, all_goals, all_lms, total_lm_position, landmarks, start_odour, num_laps


def get_lm_entry_exit(session, positions):
    '''Find data idx closest to landmark entry and exit.'''

    lm_entry_idx = []
    lm_exit_idx = []
    
    if session['num_laps'] > 1:
        search_start = 0  

        for i, (lm_start, lm_end) in enumerate(session['all_landmarks'][:-1]):  
            lm_start_idx = np.where(positions[search_start:] >= lm_start)[0][0] + search_start

            next_lm_start = session['all_landmarks'][i+1,0]
            next_lm_start_idx = np.where(positions[search_start:] >= next_lm_start)[0][0] + search_start

            if next_lm_start < lm_start:    # position reset 
                # print('Lap change')
                distance = 10 ** (int(math.log10(len(positions))) - 1) 
                height = math.floor(max(positions)/10)*10
                lap_change_idx = find_peaks(positions[search_start:], height=height, distance=distance)[0][0] + 1

                next_lm_start_idx = search_start + lap_change_idx + 1

            start_candidates = np.where(positions[search_start:next_lm_start_idx] >= lm_start)[0]
            entry_idx = start_candidates[0] + search_start
            
            end_candidates = np.where(positions[entry_idx:next_lm_start_idx] >= lm_end)[0]
            exit_idx = end_candidates[0] + entry_idx

            search_start = next_lm_start_idx 

            lm_entry_idx.append(entry_idx)
            lm_exit_idx.append(exit_idx)

        # last landmark 
        last_lm_start_idx = np.where(positions[search_start:] >= session['all_landmarks'][-1,0])[0][0] + search_start
        last_lm_end_idx = np.where(positions[search_start:] >= session['all_landmarks'][-1,1])[0]
        if len(last_lm_end_idx) != 0:
            last_lm_end_idx = last_lm_end_idx[0] + search_start
            lm_entry_idx.append(last_lm_start_idx)  
            lm_exit_idx.append(last_lm_end_idx)
        else:
            return np.array(lm_entry_idx), np.array(lm_exit_idx)  # terminate early 
    
    else:
        if (positions[0] - session['landmarks'][-1,1]) < (positions[0] - session['landmarks'][0,0]):
            search_start = np.where(positions <= session['all_landmarks'][0,0])[0][-1]  # the mouse accidentally moved backwards first
        else: 
            search_start = 0

        for lm_start in session['all_landmarks'][:,0]:
            lm_entry_idx.append(np.where(positions[search_start:] >= lm_start)[0][0] + search_start)

        for lm_end in session['all_landmarks'][:,1]:
            lm_exit_idx.append(np.where(positions[search_start:] <= lm_end)[0][-1] + search_start)

    return np.array(lm_entry_idx), np.array(lm_exit_idx)


def load_nidaq_behaviour_data(sess_data_path):
    '''Load behaviour data from NIDAQ logging - after barcode alignment.'''

    nidaq_data = np.load(os.path.join(sess_data_path, 'behaviour_data.npz'))

    return nidaq_data


def load_vr_behaviour_data(sess_data_path):
    '''Load VR data from position_log.csv and config.yaml files.'''
    
    position_data_dir = [d for d in os.listdir(os.path.join(sess_data_path, 'behav')) if d.isdigit() and len(d) == 6][0]
    VR_data = pd.read_csv(os.path.join(sess_data_path, 'behav', position_data_dir, 'position_log.csv'))

    config_file = os.path.join(sess_data_path, 'behav', position_data_dir, 'config.yaml')
    with open(config_file, 'r') as fd:
        options = yaml.load(fd, Loader=yaml.SafeLoader)  

    return VR_data, options


def get_landmark_categories(sequence, num_landmarks, session):
    '''Find the landmarks in the entire session that belong to goals, non-goals and test.'''

    session = get_landmark_ids(sequence, num_landmarks, session)

    # Get the landmarks that belong to each condition  
    goals_idx = np.where(np.isin(session['all_lms'], session['goal_landmark_id']))[0]
    non_goals_idx = np.where(np.isin(session['all_lms'], session['non_goal_landmark_id']))[0]
    test_idx = np.where(np.isin(session['all_lms'], session['test_landmark_id']))[0] if session['test_landmark_id'] is not None else None
    
    session['goals_idx'] = goals_idx
    session['non_goals_idx'] = non_goals_idx
    session['test_idx'] = test_idx

    return session


def get_landmark_ids(sequence, num_landmarks, session):
    '''Define which landmarks belong to goals, non-goals and test.'''

    if num_landmarks == 10:     # T5 and T6
        if sequence == 'ABAB':
            goal_landmark_id = np.array([1, 3, 5, 7])
            test_landmark_id = 9
        elif sequence == 'AABB':  
            goal_landmark_id = np.array([0, 1, 4, 5])
            test_landmark_id = np.array([8, 9])
        non_goal_landmark_id = np.setxor1d(np.arange(0,num_landmarks), np.append(goal_landmark_id, test_landmark_id))
 
    elif num_landmarks == 2:    # T3 and T4
        lms = np.unique(session['all_lms'])
        goal_landmark_id = session['all_lms'][session['goal_idx'][0]]
        non_goal_landmark_id = np.setdiff1d(lms, goal_landmark_id)[0]
        test_landmark_id = None

    session['goal_landmark_id'] = goal_landmark_id
    session['non_goal_landmark_id'] = non_goal_landmark_id
    session['test_landmark_id'] = test_landmark_id

    return session


def get_landmark_category_rew_idx(sequence, num_landmarks, session, VR_data, nidaq_data):
    '''Find indices also in non-goal landmarks corresponding to the same time after landmark entry as mean reward time lag.'''
    
    reward_idx = get_rewards(VR_data, nidaq_data, session, print_output=True)

    rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx = get_landmark_category_entries(VR_data, nidaq_data, sequence, num_landmarks, session)
    
    # Calculate time lag between landmark entry and reward delivery
    rew_time_lag = np.round(np.mean(reward_idx - rew_lm_entry_idx))
    print('Reward time lag from lm entry: ', rew_time_lag)

    # Find where reward would be on average if these landmarks were rewarded
    miss_rew_idx = miss_lm_entry_idx + rew_time_lag
    nongoal_rew_idx = nongoal_lm_entry_idx + rew_time_lag  
    test_rew_idx = test_lm_entry_idx + rew_time_lag

    session['rew_time_lag'] = rew_time_lag
    session['reward_idx'] = reward_idx
    session['miss_rew_idx'] = miss_rew_idx
    session['nongoal_rew_idx'] = nongoal_rew_idx
    session['test_rew_idx'] = test_rew_idx

    return session


def get_imag_rew_idx(nidaq_data, session, lm_idx):
    '''Find indices after landmark entry where reward would be expected.'''
    
    lm_entry_idx, _ = get_lm_entry_exit(session, positions=nidaq_data['position'])

    lm_entry_idx = np.array([lm_entry_idx[i] for i in lm_idx])

    imag_rew_idx = lm_entry_idx + session['rew_time_lag']

    return imag_rew_idx
    

def get_landmark_category_entries(VR_data, nidaq_data, sequence, num_landmarks, session):
    '''Find the indices of landmark entry for different types of landmarks: rewarded, miss, non-goal, test.'''
    
    lm_entry_idx, _ = get_lm_entry_exit(session, positions=nidaq_data['position'])

    # Find category for each landmark 
    session = get_landmark_categories(sequence, num_landmarks, session)

    # Find the rewarded landmarks 
    session = get_rewarded_landmarks(VR_data, nidaq_data, session)

    # Find landmark entry indices for each landmark category
    rew_lm_entry_idx = [lm_entry_idx[i] for i in session['rewarded_landmarks']]
    miss_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['goals_idx'] if i not in session['rewarded_landmarks']])
    nongoal_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['non_goals_idx']])
    test_lm_entry_idx = np.array([lm_entry_idx[i] for i in session['test_idx']]) if session['test_idx'] is not None else np.array([])

    assert len(rew_lm_entry_idx) + len(miss_lm_entry_idx) + len(nongoal_lm_entry_idx) + len(test_lm_entry_idx) == len(session['all_lms']), 'Some landmarks have not been considered.'

    return rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx


def get_rewarded_landmarks(VR_data, nidaq_data, session):
    '''Find the indices of rewarded (lick-triggered) landmarks.'''

    reward_idx = get_rewards(VR_data, nidaq_data, session, print_output=False)
    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session, positions=nidaq_data['position'])

    # Find rewarded landmarks 
    reward_positions = nidaq_data['distance'][reward_idx]  # using flattened position array 

    rewarded_landmarks = [i for i, (start, end) in enumerate(zip(np.floor(nidaq_data['distance'][lm_entry_idx]), np.ceil(nidaq_data['distance'][lm_exit_idx]))) 
                            if np.any((np.ceil(reward_positions) >= start) & (np.floor(reward_positions) <= end))] 
    
    session['rewarded_landmarks'] = rewarded_landmarks

    return session


def get_rewards(VR_data, nidaq_data, session, print_output=False):
    '''Find the indices of lick-triggered rewards in the nidaq logging file.'''

    # Find different types of rewards from VR data
    rewards_VR, assistant_reward_idx, manual_reward_idx = get_VR_rewards(VR_data)
    all_rewards_VR = np.sort(np.concatenate([rewards_VR, assistant_reward_idx, manual_reward_idx]))

    # Find rewards in NIDAQ data
    reward_idx = np.where(nidaq_data['rewards'] == 1)[0]  
    rewards_to_remove = []

    for r, rew in enumerate(all_rewards_VR):
        if (rew in assistant_reward_idx) or (rew in manual_reward_idx):
            rewards_to_remove.append(r)

    reward_idx = np.delete(reward_idx, rewards_to_remove)

    # Confirm number of rewards makes sense
    if session['all_landmarks'][-1,1] < nidaq_data['position'][reward_idx[-1]]:  # ensure mouse has left last rewarded landmark 
        reward_idx = reward_idx[0:-1]  
    num_rewards = len(reward_idx)  

    if print_output:
        print('Total rewards considered here: ', num_rewards)
        print('Total rewards not considered here: ', len(rewards_to_remove))
        print('Total assistant and manual rewards: ', len(assistant_reward_idx) + len(manual_reward_idx))

    return reward_idx


def get_VR_rewards(VR_data):
    '''Find different types of rewards from VR data.'''
    # rewards_root_VR = np.where(VR_data['Event'] == 'rewarded')[0]
    # rewards_VR = VR_data['Index'][rewards_root_VR].values
    rewards_VR = np.where(VR_data['Event'] == 'rewarded')[0]

    # assistant_reward_root_idx = np.where(VR_data['Event'] == 'assist-rewarded')[0]
    # assistant_reward_idx = VR_data['Index'][assistant_reward_root_idx].values
    assistant_reward_idx = np.where(VR_data['Event'] == 'assist-rewarded')[0]

    # manual_reward_root_idx = np.where(VR_data['Event'] == 'manually-rewarded')[0]
    # manual_reward_idx = VR_data['Index'][manual_reward_root_idx].values
    manual_reward_idx = np.where(VR_data['Event'] == 'manually-rewarded')[0]

    return rewards_VR, assistant_reward_idx, manual_reward_idx


def get_licks(nidaq_data, session, print_output=False):
    '''Find the indices of licks in the nidaq logging file.'''

    # Find licks in NIDAQ data
    lick_idx = np.where(nidaq_data['licks'] == 1)[0]  

    # Confirm number of rewards makes sense
    if session['all_landmarks'][-1,1] < nidaq_data['position'][lick_idx[-1]]:  # TODO ensure mouse has left last licked landmark 
        lick_idx = lick_idx[0:-1]  
    num_licks = len(lick_idx)  

    session['lick_idx'] = lick_idx
    if print_output:
        print('Total licks considered here: ', num_licks)
        
    return session


def get_lick_types(session, VR_data, nidaq_data):
    """
    Give an ID to each lick type:
    1: licks inside goal landmarks & rewarded (hit)
    2: licks inside goal landmarks & not rewarded (miss)
    3: licks inside non-goal landmarks (false alarm)
    4: licks inside test landmark
    5: licks before goal landmarks 
    6: licks before non-goal landmarks 
    7: licks before test landmark
    8: imaginary lick inside goal (miss)
    9: imaginary lick inside non-goal 
    10: imaginary lick inside test

    Returns:
    -------
    licks: Dict where the key corresponds to the lick ID
    """

    # Get landmark entry and exit indices 
    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session, nidaq_data['position'])

    # Get lick indices
    session = get_licks(nidaq_data, session)
    lick_idx = session['lick_idx']

    # Get rewarded landmarks
    session = get_rewarded_landmarks(VR_data, nidaq_data, session)

    # Collect all licks 
    licks = {id: {} for id in range(1,11)}

    for id in range(1,11):
        collected_licks = []  
        
        if id == 1:
            collected_licks = [
                lick_idx[(lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['rewarded_landmarks']
            ]

        elif id == 2:
            collected_licks = [
                lick_idx[(lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['goals_idx'] and i not in session['rewarded_landmarks']
            ]

        elif id == 3:
            collected_licks = [
                lick_idx[(lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['non_goals_idx']
            ]

        elif id == 4:
            if session['test_idx'] is not None:
                collected_licks = [
                    lick_idx[(lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i])]
                    for i in range(len(lm_entry_idx)) if i in session['test_idx']
                ]
            else:
                collected_licks = [] 

        elif id == 5:
            if 0 in session['goals_idx']:
                collected_licks.append(lick_idx[lick_idx < lm_entry_idx[0]])
            collected_licks += [
                lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])]
                for i in range(len(lm_entry_idx) - 1) if i + 1 in session['goals_idx']
            ]

        elif id == 6:
            if 0 in session['non_goals_idx']:
                collected_licks.append(lick_idx[lick_idx < lm_entry_idx[0]])
            collected_licks += [
                lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])]
                for i in range(len(lm_entry_idx) - 1) if i + 1 in session['non_goals_idx']
            ]

        elif id == 7:
            if session['test_idx'] is not None:
                if 0 in session['test_idx']:
                    collected_licks.append(lick_idx[lick_idx < lm_entry_idx[0]])  
                collected_licks += [
                    lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])]
                    for i in range(len(lm_entry_idx) - 1) if i + 1 in session['test_idx']
                ]
            else: 
                collected_licks = []

        elif id == 8:
            collected_licks = [
                session['miss_rew_idx'][(session['miss_rew_idx'] >= lm_entry_idx[i]) & (session['miss_rew_idx'] <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['goals_idx'] and i not in session['rewarded_landmarks'] 
                and not np.any((lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i]))
            ]

        elif id == 9:
            collected_licks = [
                session['nongoal_rew_idx'][(session['nongoal_rew_idx'] >= lm_entry_idx[i]) & (session['nongoal_rew_idx'] <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['non_goals_idx']
                and not np.any((lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i]))
            ]

        elif id == 10:
            collected_licks = [
                session['test_rew_idx'][(session['test_rew_idx'] >= lm_entry_idx[i]) & (session['test_rew_idx'] <= lm_exit_idx[i])]
                for i in range(len(lm_entry_idx)) if i in session['test_idx']
                and not np.any((lick_idx >= lm_entry_idx[i]) & (lick_idx <= lm_exit_idx[i]))
            ]

        if collected_licks:
            licks[id] = np.concatenate(collected_licks).astype(int)
        else:
            licks[id] = np.array([], dtype=int)
            
    return licks


def get_first_licks(session, VR_data, nidaq_data):
    """
    Get the first lick from each type in a block of licks

    Returns:
    -------
    first_licks: Dict where the key corresponds to the lick ID
    """

    # Get landmark entry and exit indices 
    lm_entry_idx, lm_exit_idx = get_lm_entry_exit(session, nidaq_data['position'])

    # Get lick indices
    session = get_licks(nidaq_data, session)
    lick_idx = session['lick_idx']

    # Get lick ids
    licks = get_lick_types(session, VR_data, nidaq_data)

    # Find first licks for each type 
    first_licks = {}

    for id in range(1,11):
        if id == 1:
            first_licks[id] = np.array([lick_idx[(lick_idx >= entry) & (lick_idx <= exit)][0]
                    for i, (entry, exit) in enumerate(zip(lm_entry_idx, lm_exit_idx))
                    if i in session['rewarded_landmarks'] and np.any((lick_idx >= entry) & (lick_idx <= exit))])
            
        elif id == 2:
            first_licks[id] = np.array([lick_idx[(lick_idx >= entry) & (lick_idx <= exit)][0]
                    for i, (entry, exit) in enumerate(zip(lm_entry_idx, lm_exit_idx))
                    if i in session['goals_idx'] and i not in session['rewarded_landmarks'] and np.any((lick_idx >= entry) & (lick_idx <= exit))])
            
        elif id == 3:
            first_licks[id] = np.array([lick_idx[(lick_idx >= entry) & (lick_idx <= exit)][0]
                    for i, (entry, exit) in enumerate(zip(lm_entry_idx, lm_exit_idx))
                    if i in session['non_goals_idx'] and np.any((lick_idx >= entry) & (lick_idx <= exit))])
            
        elif id == 4:
            if session['test_idx'] is not None:
                first_licks[id] = np.array([lick_idx[(lick_idx >= entry) & (lick_idx <= exit)][0]
                        for i, (entry, exit) in enumerate(zip(lm_entry_idx, lm_exit_idx))
                        if i in session['test_idx'] and np.any((lick_idx >= entry) & (lick_idx <= exit))])
            else:
                first_licks[id] = []
            
        elif id == 5:
            licks5 = []
            if 0 in session['goals_idx']:
                if np.any(lick_idx < lm_entry_idx[0]):
                    licks5.append(lick_idx[lick_idx < lm_entry_idx[0]][0])
            licks5 += [
                lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])][0]
                for i in range(len(lm_entry_idx) - 1)
                if i + 1 in session['goals_idx'] and np.any((lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1]))
            ]
            first_licks[id] = np.array(licks5)

        elif id == 6:
            licks6 = []
            if 0 in session['non_goals_idx']:
                if np.any(lick_idx < lm_entry_idx[0]):
                    licks6.append(lick_idx[lick_idx < lm_entry_idx[0]][0])
            licks6 += [
                lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])][0]
                for i in range(len(lm_entry_idx) - 1)
                if i + 1 in session['non_goals_idx'] and np.any((lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1]))
            ]
            first_licks[id] = np.array(licks6)

        elif id == 7:
            if session['test_idx'] is not None:
                licks7 = []
                if 0 in session['test_idx']:
                    if np.any(lick_idx < lm_entry_idx[0]):
                        licks7.append(lick_idx[lick_idx < lm_entry_idx[0]][0])
                licks7 += [
                    lick_idx[(lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1])][0]
                    for i in range(len(lm_entry_idx) - 1)
                    if i + 1 in session['test_idx'] and np.any((lick_idx > lm_exit_idx[i]) & (lick_idx < lm_entry_idx[i + 1]))
                ]
                first_licks[id] = np.array(licks7)
            else:
                first_licks[id] = []

        elif id == 8 or id == 9 or id == 10: 
            first_licks[id] = licks[id]

    # Concatenate all first licks together and sort them by data index
    all_first_licks = np.sort(np.concatenate([first_licks[id] for id in range(1,11)]))
    print('Number of licks considered:', len(all_first_licks))

    # Confirm the licks and imaginary licks in landmarks make sense
    all_reward_idx = np.sort(np.concatenate([session['reward_idx'], session['miss_rew_idx'], session['nongoal_rew_idx'], session['test_rew_idx']]))

    assert len (all_reward_idx) == (len(first_licks[1]) + len(first_licks[2]) + len(first_licks[3]) + \
                                    len(first_licks[4]) + len(first_licks[8]) + len(first_licks[9]) + \
                                        len(first_licks[10])), 'Some licks or rewards are missing...'
    return first_licks, all_first_licks


def threshold_nidaq_licks(nidaq_data, session):
    """Threshold which licks are considered based on the animal's speed."""
    
    lick_threshold = session['lick_threshold']
    threshold_licks = session['lick_idx'][np.where(nidaq_data['speed'][session['lick_idx']] < lick_threshold)[0]]

    session['thresholded_lick_idx'] = threshold_licks

    return session


def get_lick_rate(nidaq_data, session):
    """Get lick rate per frame as a sliding window - similar to parse_session_functions.calculate_lick_rate"""
    
    # Threshold licks
    session = threshold_nidaq_licks(nidaq_data, session)

    # Calculate lick rate as the mean number of licks over sliding window
    window = 100 # frames
    lick_rate = np.zeros(len(nidaq_data['position']))
    for i in range(len(nidaq_data['position'])-window):
        if 'thresholded_lick_idx' in session:
            lick_num = len(np.where((session['thresholded_lick_idx'] > i) & (session['thresholded_lick_idx'] < i+window))[0])
        else:
            lick_num = len(np.where((session['lick_idx'] > i) & (session['lick_idx'] < i+window))[0])
        lick_rate[i] = lick_num / window
    
    session['frame_lick_rate'] = lick_rate

    return session


def get_smoothed_lick_rate(nidaq_data, session):
    """Get lick rate per frame as a sliding window - similar to parse_session_functions.calculate_lick_rate"""
    
    # Threshold licks
    session = threshold_nidaq_licks(nidaq_data, session)

    # Calculate smoothed lick rate 
    binary_licks = np.zeros_like(nidaq_data['position'], dtype=int)
    binary_licks[session['thresholded_lick_idx']] = 1

    lick_rate = gaussian_filter1d(binary_licks.astype(float), sigma=1.5)
    lick_rate = lick_rate.reshape(1,-1)

    session['smooth_lick_rate'] = lick_rate

    return session


def get_event_lick_rate(session, event_idx, time_around=(-1,3), funcimg_frame_rate=45):
    """Get lick rate per frame as a smoothed sliding window around an event"""
    
    # Handle single int input as symmetric window
    if isinstance(time_around, (int, float)):
        start_time = -time_around
        end_time = time_around
    elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
        start_time, end_time = time_around
    else:
        raise ValueError("time_around must be a single number or a tuple/list of (start, end)")

    start_frames = int(np.floor(start_time * funcimg_frame_rate))
    end_frames = int(np.ceil(end_time * funcimg_frame_rate))

    # Get indices for each event
    window = np.arange(start_frames, end_frames)
    window_indices = np.add.outer(event_idx, window).astype(int)

    # Find licks within this window
    binary_licks = np.zeros_like(window_indices, dtype=int)
    binary_licks = np.isin(window_indices, session['lick_idx']).astype(int)

    # Get smoothed lick rate 
    event_lick_rate = np.empty_like(window_indices, dtype=float)
    for i in range(window_indices.shape[0]):
        event_lick_rate[i,:] = gaussian_filter1d(binary_licks[i,:].astype(float), sigma=1.5)
    
    return event_lick_rate


def get_lm_lick_rate(nidaq_data, session):
    """Get lick rate per frame bin as the mean per bin for each landmark"""
    
    # Get all datapoints within landmarks
    session = get_data_lm_idx(nidaq_data, session)

    lm_lick_rate = {}

    # Create a binary lick map for the entire session
    binary_licks = np.zeros(len(nidaq_data['position']))
    binary_licks[session['thresholded_lick_idx']] = 1

    for lap in range(session['num_laps']):
        for lm in range(len(session['all_lms'])):
            key = (lap, lm)

            # datapoints within landmarks for each lap 
            lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

            # binary licks within landmark
            lm_licks = binary_licks[lm_idx[0]:lm_idx[-1]+1]
            
            # calculate lick rate within each landmark (mean in each bin)
            lm_lick_rate[key], _, _ = stats.binned_statistic(lm_idx, lm_licks, bins=16)

    session['lm_lick_rate'] = lm_lick_rate

    return session


def get_binned_lick_rate(nidaq_data, session):  # TODO
    """Get lick rate per frame bin as the mean per bin for each landmark and the gray zones before"""
    
    # Get all datapoints within landmarks
    session = get_data_lm_idx(nidaq_data, session)

    lm_lick_rate = {}

    # Create a binary lick map for the entire session
    binary_licks = np.zeros(len(nidaq_data['position']))
    binary_licks[session['thresholded_lick_idx']] = 1

    for lap in range(session['num_laps']):
        for lm in range(len(session['all_lms']) * 2):
            key = (lap, lm)

            # datapoints within landmarks for each lap 
            lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

            # binary licks within landmark
            lm_licks = binary_licks[lm_idx[0]:lm_idx[-1]+1]
            
            # calculate lick rate within each landmark (mean in each bin)
            lm_lick_rate[key], _, _ = stats.binned_statistic(lm_idx, lm_licks, bins=16)

    session['lm_lick_rate'] = lm_lick_rate

    return session


def get_binary_lick_map(nidaq_data, session):
    """Create a binary map of licked landmarks - similar to parse_session_functions.get_licked_lms"""
    
    # Get all datapoints within landmarks
    session = get_data_lm_idx(nidaq_data, session)

    licked_lms = np.zeros((session['num_laps'], len(session['all_lms'])))

    for lap in range(session['num_laps']):
        for lm in range(len(session['all_lms'])):

            # datapoints within landmarks for each lap 
            lm_idx = np.where(session['data_lm_idx'] == lm+1)[0]

            # Find all licks within the landmark
            if 'thresholded_lick_idx' in session:
                target_licks = np.intersect1d(lm_idx, session['thresholded_lick_idx'])
            else:
                target_licks = np.intersect1d(lm_idx, session['lick_idx'])
            if len(target_licks) > 0:
                licked_lms[lap,lm] = 1
            else:
                licked_lms[lap,lm] = 0

    session['binary_licked_lms'] = licked_lms

    return session


def get_licks_per_lap(session):
    #save lick indices for each lap in a dictionary
    lick_positions = {}
    for i in range(session['num_laps']):
        lap_ix = np.where(session['lap_idx'] == i)[0]
        licks_per_lap_ix = np.intersect1d(lap_ix, session['thresholded_licks'])
        lick_positions[i] = session['position'][licks_per_lap_ix]

    session['licks_per_lap'] = lick_positions

    return session


def get_data_lm_idx(nidaq_data, session):
    """Get the landmark id of every data entry - similar to parse_session_functions.get_lm_idx"""
    
    # Find landmark entry and exit idx
    lm_entry, lm_exit = get_lm_entry_exit(session, nidaq_data['position'])

    # Find datapoints within a landmark
    lm_idx = np.zeros(len(nidaq_data['position']))
    for i in range(len(session['all_lms'])):
        lm_idx[lm_entry[i]:lm_exit[i]+1] = i+1

    session['data_lm_idx'] = lm_idx

    return session


def get_position_info(VR_data):
    '''Find position, speed, total distance, times from VR data.'''
    position_idx = np.where(VR_data['Position'] > -1)[0]
    
    times = VR_data['Time'][position_idx].values

    position = VR_data['Position'][position_idx].values 
    total_dist = VR_data['TotalRunDistance'][position_idx].values #- np.array(options['flip_tunnel']['margin_start'])

    if 'Speed' not in VR_data.keys():
        speed = np.diff(total_dist)/np.diff(times)
        speed = np.append(speed, speed[-1])
    else:
        speed = VR_data['Speed'][position_idx].values
    
    return times, position, speed, total_dist