import numpy as np 
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pandas as pd
import yaml


def get_psth(data, neurons, event_idx, time_around=1, funcimg_frame_rate=45):

    num_neurons = len(neurons)
    num_events = len(event_idx)

    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = 2*time_window

    window_indices = np.add.outer(event_idx, np.arange(-time_window, time_window)).astype(int)  

    psth = np.zeros((num_neurons, num_events, num_timebins))
    for n, neuron in enumerate(neurons):
        psth[n, :, :] = data[neuron, window_indices]

    # Average PSTH for all events
    average_psth = np.zeros([num_neurons, num_timebins])
    average_psth = np.mean(psth, axis=1)

    return psth, average_psth


def plot_avg_psth(average_psth, neurons, event='reward', zscoring=True, time_around=1, funcimg_frame_rate=45, save_psth=False, savepath='', filename=''):

    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    num_neurons = len(neurons)

    # Sort cells according to firing around event
    sortidx = np.argsort(np.argmax(average_psth, axis=1))

    data = average_psth.copy()
    if zscoring:
        data = stats.zscore(data, axis=1)

    fig, ax = plt.subplots(figsize=(3,4))
    im = ax.imshow(data[sortidx, :], aspect='auto')
    ax.vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
    ax.set_xlabel('Time')
    ax.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
    if time_around == int(time_around):
        xticklabels = [int(-time_around), 0, int(time_around)]
    else:
        xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
    ax.set_xticklabels(xticklabels)


    ax.set_ylabel('Neuron')
    ax.set_yticks([-0.5, num_neurons-0.5])
    ax.set_yticklabels([0, num_neurons])
    fig.suptitle(f'{event} PSTH')

    cbar = fig.colorbar(im, ax=ax)
    vmin, vmax = im.get_clim()
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=2, fontsize=8)
    plt.tight_layout()

    if save_psth:
        plt.savefig(os.path.join(savepath, f'{filename}.png'))


def split_psth(psth, neurons, event_idx, event='reward', zscoring=True, time_around=1, funcimg_frame_rate=45):

    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    num_neurons = len(neurons)
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
    
    vmin = min(np.min(sorting_data), np.min(testing_data))
    vmax = max(np.max(sorting_data), np.max(testing_data))

    sortidx = np.argsort(np.argmax(sorting_data[:, :], axis=1))

    # Plotting 
    fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])  # third slot for colorbar

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    cax = fig.add_subplot(gs[2])

    # fig, ax = plt.subplots(1, 2, figsize=(6,4), sharey=True)
    # ax = ax.ravel()

    im0 = ax0.imshow(sorting_data[sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
    ax0.vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
    ax0.set_xlabel('Time')
    ax0.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
    ax0.set_xticklabels([int(-time_around), 0, int(time_around)])
    ax0.set_title(f'Sorting trials')

    im1 = ax1.imshow(testing_data[sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
    ax1.vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
    ax1.set_xlabel('Time')
    ax1.set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
    ax1.set_xticklabels([int(-time_around), 0, int(time_around)])
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


def get_tuned_neurons(average_psth, neurons, event='reward', time_around=1, funcimg_frame_rate=45, plot_neurons=True):
    # Statistics to find neurons tuned to an event e.g. reward, lick, landmark entry etc.
    # TODO: bootstrapping / permutation test instead? 

    # Mann–Whitney U test comparing the 1-s period just before stimulus onset to the 1-s period directly after stimulus onset. 

    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    num_neurons = len(neurons)

    before_event_firing = average_psth[:, 0:time_window]
    after_event_firing = average_psth[:, time_window:]
    # print(before_event_firing.shape, after_event_firing.shape)

    wilcoxon_stat = np.zeros((num_neurons, 1))
    wilcoxon_pval = np.zeros((num_neurons, 1))
    for n in range(num_neurons):
            wilcoxon_stat[n], wilcoxon_pval[n] = stats.wilcoxon(before_event_firing[n, :], after_event_firing[n, :]) #, method=stats.PermutationMethod(n_resamples=1000))

    tuned_neurons = np.where(wilcoxon_pval < 0.05)[0]
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


def get_tuned_neurons_shohei(DF_F, average_psth, neurons, event='reward', time_around=1, funcimg_frame_rate=45, plot_neurons=True, zscoring=True):
    # The response to an event is calculated using the mean z-scored ΔF/F calcium signal 
    # averaged over a window from 0.4 s to 1 s after event onset, baseline-subtracted using 
    # the mean z-scored ΔF/F signal during 0.5 s before event onset for each event. 
    # Neurons are classified as event-responsive if their mean response is bigger than 0.5 z-scored ΔF/F. 
    
    time_window = time_around * funcimg_frame_rate # frames
    time_before = int(np.floor(0.5 * funcimg_frame_rate))
    time_after = int(0.4 * funcimg_frame_rate)
    num_timebins = 2*time_window

    num_neurons = len(neurons)

    data = average_psth.copy()
    if zscoring:
        data = stats.zscore(np.array(data), axis=1)

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

    num_neurons = len(neurons)
    num_events = len(event_idx)

    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    window_indices = np.add.outer(event_idx, np.arange(-time_window, time_window)).astype(int)  

    psth = np.zeros((num_neurons, num_events, num_timebins))
    for n, neuron in enumerate(neurons):
        psth[n, :, :] = data[neuron, window_indices]

    # Average PSTH for all events per landmark
    average_landmark_psth = np.zeros([num_neurons, num_landmarks, num_timebins])
    for i in range(num_landmarks):
        average_landmark_psth[:, i, :] = np.mean(psth[:, i::num_landmarks, :], axis=1)

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


def plot_landmark_psth_map(average_psth, zscoring=True, sorting_lm=0, num_landmarks=10, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir='', filename=''):
    '''Plot firing maps of all selected neurons for all landmarks, sorted by specific landmark.'''

    if sorting_lm >= num_landmarks:
        raise ValueError(f'The sorting landmark should be one of the {num_landmarks} landmarks.')
    
    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    fig, ax = plt.subplots(1, 10, figsize=(15,3), sharey=True, sharex=True)
    ax = ax.ravel()

    data = average_psth.copy()
    if zscoring:
        data = stats.zscore(data, axis=1)
    
    sortidx = np.argsort(np.argmax(data[:, sorting_lm, :], axis=1))

    for i in range(num_landmarks):
        ax[i].imshow(data[sortidx, i, :], aspect='auto')
        ax[i].vlines(time_window-0.5, ymin=-0.5, ymax=data.shape[0]-0.5, color='k', linewidth=0.5)
        ax[i].set_xlabel('Time')
        ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
        ax[i].set_xticklabels([int(-time_around), 0, int(time_around)])
        ax[i].spines[['right', 'top']].set_visible(False)
        ax[i].set_title(f'{i+1}')

    ax[0].set_yticks([-0.5, data.shape[0]-0.5])
    ax[0].set_yticklabels([0, data.shape[0]])
    ax[0].set_ylabel('Neuron')
    plt.tight_layout()

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


def plot_all_sessions_goal_psth_map(all_average_psths, zscoring=True, sorting_goal=1, time_around=1, funcimg_frame_rate=45, save_plot=False, savepath='', savedir='', filename=''):
    '''Plot firing maps for all sessions and each goal, sorted by a specific goal.'''

    num_sessions = len(all_average_psths)
    num_goals = len(all_average_psths[0])  # assuming each session has same number of goals

    if num_goals == 4:
        goals = ['A','B','C','D']
    else:
        goals = ['A','B']

    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    # Copy and optionally z-score
    data = []
    for session in all_average_psths:
        session_data = {}
        for goal in session.keys():
            session_data[goal] = stats.zscore(session[goal], axis=1) if zscoring else session[goal]
        data.append(session_data)

    # Compute global vmin/vmax
    vmin = min([np.nanmin(session[goal]) for session in data for goal in session.keys()])
    vmax = max([np.nanmax(session[goal]) for session in data for goal in session.keys()])

    # Sort neurons consistently across sessions (using sorting_goal)
    sortidx = np.argsort(np.argmax(data[0][sorting_goal], axis=1))  # reference the first session for sorting

    # Set up figure
    fig, ax = plt.subplots(num_sessions, num_goals, figsize=(3*num_goals, 3*num_sessions), sharex=True, sharey=True)
    if num_sessions == 1 or num_goals == 1:
        ax = np.atleast_2d(ax)
    ax = np.array(ax)

    for s in range(num_sessions):
        for g, goal in enumerate(sorted(data[s].keys())):
            ax[s,g].imshow(data[s][goal][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
            ax[s,g].vlines(time_window-0.5, ymin=-0.5, ymax=data[s][goal].shape[0]-0.5, color='k', linewidth=0.5)
            ax[s,g].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            ax[s,g].set_xticklabels([int(-time_around), 0, int(time_around)])
            ax[s,g].spines[['right', 'top']].set_visible(False)
            if s == 0:
                ax[s,g].set_title(goals[g])
            if g == 0:
                ax[s,g].set_ylabel(f'Session {s+1}\nNeuron')

    # Add colorbar
    cbar = fig.colorbar(ax[0,0].images[0], ax=ax.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=10, fontsize=8)

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

    # Find global vmin and vmax across all conditions
    vmin = min([np.nanmin(d) for d in data])
    vmax = max([np.nanmax(d) for d in data])

    # Sort by different conditions
    for c, condition in enumerate(conditions):
        sortidx = np.argsort(np.argmax(data[c], axis=1))
        
        im = [[] for _ in range(len(conditions))]
        fig, ax = plt.subplots(1, len(conditions), figsize=(4*len(conditions),4), sharex=True, sharey=True)
        ax = ax.ravel()
        
        for i in range(len(conditions)):
            im[i] = ax[i].imshow(data[i][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)    
            ax[i].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            if time_around == int(time_around):
                xticklabels = [int(-time_around), 0, int(time_around)]
            else:
                xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
            ax[i].set_xticklabels(xticklabels)
            ax[i].spines[['right', 'top']].set_visible(False)
            ax[i].set_xlabel('Time')
            ax[i].set_title(f'{conditions[i]}', fontsize=10)
            ax[i].vlines(time_window-0.5, ymin=-0.5, ymax=num_neurons-0.5, color='k')
        
        ax[0].set_yticks([-0.5, num_neurons-0.5])
        ax[0].set_yticklabels([0, num_neurons])
        ax[0].set_ylabel('Neuron')

        cbar = fig.colorbar(im[-1], ax=ax.ravel().tolist(), shrink=0.6)
        cbar.set_ticks([vmin, vmax])
        cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
        cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=10, fontsize=8)

        plt.suptitle(f'Sorting by {condition} trials')

        if save_plot:
            output_path = os.path.join(savepath, savedir)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, f'{condition}_sorting.png'))
        plt.show()
        

def get_map_correlation(neurons, psths, average_psths, conditions, zscoring=True, reference=0, color_scheme=None, save_plot=False, savepath='', savedir='', filename=''):

    
    # Check data format
    if isinstance(average_psths, list):
        if reference > len(conditions):
            raise ValueError('The reference data should be within the range of input average PSTHs.')
    
        data = [average_psths[c] for c in range(len(conditions))]
        if zscoring:
            data = stats.zscore(np.array(data), axis=2)

    elif isinstance(average_psths, dict):
        if reference not in average_psths:
            raise ValueError(f'The reference data should be one of the keys of the data.')
    
        data = average_psths.copy()
        if zscoring:
            for goal in data.keys():
                data[goal] = stats.zscore(data[goal], axis=1)
   
    if isinstance(data, list):
        data_indices = np.arange(0, len(conditions))
    elif isinstance(average_psths, dict):
        data_indices = data.keys()

    corrs = [[] for c in data_indices]

    # Random half trials 
    num_sort_trials = np.floor(psths[reference].shape[1]/2).astype(int)
    event_array = np.arange(0, psths[reference].shape[1])

    random_rew_sort = np.random.choice(event_array, num_sort_trials, replace=False)  # used for sorting
    random_rew_test = np.setdiff1d(event_array, random_rew_sort)  # used for testing

    sorting_data = np.mean(psths[reference][:, random_rew_sort, :], axis=1)
    testing_data = np.mean(psths[reference][:, random_rew_test, :], axis=1)

    # Calculate correlations
    

    for c in data_indices:
        for n in range(len(neurons)):

            if c == reference:
                if np.all(np.isfinite(sorting_data[n])) and np.all(np.isfinite(sorting_data[n])):
                    r, _ = stats.pearsonr(sorting_data[n], testing_data[n])
                    corrs[c].append(r)
                else:
                    corrs[c].append(np.nan)
            else:
                if np.all(np.isfinite(data[reference][n])) and np.all(np.isfinite(data[c][n])):
                    r, _ = stats.pearsonr(data[reference][n], data[c][n])
                    corrs[c].append(r)
                else:
                    corrs[c].append(np.nan)
    
    # Convert to numpy arrays
    for c in range(len(conditions)):
        corrs[c] = np.array(corrs[c])

    # === Plotting ===
    labels = []
    for i, cond in enumerate(conditions):
        labels.append(f"{cond}\nvs\n{conditions[reference]}")

    # Compute mean and SEM for each condition's correlations
    bar_data = []
    sem_data = []
    for c in corrs:
        if np.all(np.isnan(c)):
            bar_data.append(0.0)          # Placeholder bar height
            sem_data.append(0.0)          # No error bar
        else:
            bar_data.append(np.nanmean(c))
            sem_data.append(stats.sem(c[~np.isnan(c)]) if np.sum(~np.isnan(c)) > 1 else 0)

    # bar_data = [np.nanmean(c) for c in corrs]
    # sem_data = [stats.sem(c) if len(c) > 1 else 0 for c in corrs]

    # Fallback color scheme if none is given
    if color_scheme is None:
        import seaborn as sns
        color_scheme = sns.color_palette("Set2", len(corrs))
    
    fig, ax = plt.subplots(figsize=(len(corrs)+1, 4))
    ax.bar(labels, bar_data, yerr=sem_data, capsize=3, color=color_scheme[:len(corrs)])
    ax.set_ylabel('Mean correlation')
    ax.set_title('Per-neuron PSTH correlations')
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, filename + '.png'))
    plt.show()

    return corrs


def load_vr_session_info(sess_data_path, VR_data=None, options=None):
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
    position_idx = np.where(VR_data['Position'] > -1)[0]
    position = VR_data['Position'][position_idx].values 
    times = VR_data['Time'][position_idx].values

    goals = np.array(options['flip_tunnel']['goals']) #- np.array(options['flip_tunnel']['margin_start'])
    landmarks = np.array(options['flip_tunnel']['landmarks']) #- np.array(options['flip_tunnel']['margin_start'])
    tunnel_length = options['flip_tunnel']['length']

    total_dist = VR_data['TotalRunDistance'][position_idx].values #- np.array(options['flip_tunnel']['margin_start'])
    num_laps = np.ceil([total_dist.max()/position.max()])
    num_laps = num_laps.astype(int)[0]

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


def get_lm_entry_exit(num_laps, all_lms, landmarks, positions):
    '''Find data idx closest to landmark entry and exit.'''

    lm_entry_idx = []
    lm_exit_idx = []

    if num_laps > 1:
        search_start = 0  

        for i, (lm_start, lm_end) in enumerate(landmarks[0:len(all_lms)]):
            idx_candidates = np.where((positions[search_start:] >= lm_start) & (positions[search_start:] <= lm_end))[0]
            if len(idx_candidates) > 0:
                lm_entry_idx.append(idx_candidates[0] + search_start)
                lm_exit_idx.append(idx_candidates[-1] + search_start)  # TODO: confirm
                search_start += idx_candidates[0] 
            else:
                print(f"Warning: no match found for landmark {i} with bounds {lm_start}-{lm_end}")
                lm_entry_idx.append(None)

    else:
        for lm_start in landmarks[0:len(all_lms)][:,0]:
            lm_entry_idx.append(np.where(positions >= lm_start)[0][0])
            # lm_entry_idx2.append(int(np.argmin(np.abs(positions - lm_start)))) 

        for lm_end in landmarks[0:len(all_lms)][:,1]:
            # lm_exit_idx.append(int(np.argmin(np.abs(positions - lm_end))))
            lm_exit_idx.append(np.where(positions <= lm_end)[0][-1])

    return lm_entry_idx, lm_exit_idx


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


def get_landmark_categories(sequence, num_landmarks, all_lms):
    '''Define which landmarks belong to goals, non-goals and test.'''

    if sequence == 'ABAB':
        goal_landmark_id = [1, 3, 5, 7]
        test_landmark_id = 9
    elif sequence == 'AABB':  # TODO
        goal_landmark_id = [0, 1, 4, 5]
        test_landmark_id = 8
    non_goal_landmark_id = np.setxor1d(np.arange(0,num_landmarks), np.append(goal_landmark_id, test_landmark_id))

    # Get the landmarks that belong to each condition
    goals_idx = np.where(np.isin(all_lms, goal_landmark_id))[0]
    non_goals_idx = np.where(np.isin(all_lms, non_goal_landmark_id))[0]
    test_idx = np.where(np.isin(all_lms, test_landmark_id))[0]

    return goals_idx, non_goals_idx, test_idx


def get_landmark_category_rew_idx(sequence, num_landmarks, landmarks, all_lms, num_laps, VR_data, nidaq_data):
    '''Find indices also in non-goal landmarks corresponding to the same time after landmark entry as mean reward time lag.'''
    
    reward_idx = get_rewards(VR_data, nidaq_data, print_output=True)

    rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx = get_landmark_category_entries(VR_data, nidaq_data, sequence, num_landmarks, all_lms, num_laps, landmarks)
    
    # Calculate time lag between landmark entry and reward delivery
    rew_time_lag = np.round(np.mean(reward_idx - rew_lm_entry_idx))
    print('Reward time lag from lm entry: ', rew_time_lag)

    # Find where reward would be on average if these landmarks were rewarded
    miss_rew_idx = miss_lm_entry_idx + rew_time_lag
    nongoal_rew_idx = nongoal_lm_entry_idx + rew_time_lag  
    test_rew_idx = test_lm_entry_idx + rew_time_lag

    return rew_time_lag, reward_idx, miss_rew_idx, nongoal_rew_idx, test_rew_idx


def get_landmark_category_entries(VR_data, nidaq_data, sequence, num_landmarks, all_lms, num_laps, landmarks):
    '''Find the indices of landmark entry for different types of landmarks: rewarded, miss, non-goal, test.'''
    
    lm_entry_idx, _ = get_lm_entry_exit(num_laps, all_lms, landmarks, positions=nidaq_data['position'])

    # Find category for each landmark 
    goals_idx, non_goals_idx, test_idx = get_landmark_categories(sequence, num_landmarks, all_lms)

    # Find the rewarded landmarks 
    rewarded_landmarks = get_rewarded_landmarks(VR_data, nidaq_data, landmarks, all_lms)

    # Find landmark entry indices for each landmark category
    rew_lm_entry_idx = [lm_entry_idx[i] for i in rewarded_landmarks]
    miss_lm_entry_idx = np.array([lm_entry_idx[i] for i in goals_idx if i not in rewarded_landmarks])
    nongoal_lm_entry_idx = np.array([lm_entry_idx[i] for i in non_goals_idx])
    test_lm_entry_idx = np.array([lm_entry_idx[i] for i in test_idx])

    assert len(rew_lm_entry_idx) + len(miss_lm_entry_idx) + len(nongoal_lm_entry_idx) + len(test_lm_entry_idx) == len(all_lms), 'Some landmarks have not been considered.'

    return rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx


def get_rewarded_landmarks(VR_data, nidaq_data, landmarks, all_lms):
    '''Find the indices of rewarded (lick-triggered) landmarks.'''

    reward_idx = get_rewards(VR_data, nidaq_data, print_output=False)

    # Find rewarded landmarks 
    landmark_positions = landmarks[:][0:len(all_lms)]
    reward_positions = nidaq_data['position'][reward_idx]

    rewarded_landmarks = [i for i, (start, end) in enumerate(landmark_positions) 
                        if np.any((reward_positions >= start) & (reward_positions <= end))]  # TODO: what is wrong with the last reward? 
    
    return rewarded_landmarks


def get_rewards(VR_data, nidaq_data, print_output=False):
    '''Find the indices of rewards in the nidaq logging file.'''

    # Find different types of rewards 
    rewards_root_VR = np.where(VR_data['Event'] == 'rewarded')[0]
    rewards_VR = VR_data['Index'][rewards_root_VR].values

    assistant_reward_root_idx = np.where(VR_data['Event'] == 'assist-rewarded')[0]
    assistant_reward_idx = VR_data['Index'][assistant_reward_root_idx].values

    manual_reward_root_idx = np.where(VR_data['Event'] == 'manually-rewarded')[0]
    manual_reward_idx = VR_data['Index'][manual_reward_root_idx].values

    all_rewards_VR = np.sort(np.concatenate([rewards_VR, assistant_reward_idx, manual_reward_idx]))

    reward_idx = np.where(nidaq_data['rewards'] == 1)[0]  
    rewards_to_remove = []

    for r, rew in enumerate(all_rewards_VR):
        if (rew in assistant_reward_idx) or (rew in manual_reward_idx):
            rewards_to_remove.append(r)

    reward_idx = np.delete(reward_idx, rewards_to_remove)

    # Confirm number of rewards makes sense
    reward_idx = reward_idx[0:-1]  # TODO: Deal with last reward...
    num_rewards = len(reward_idx)  

    if print_output:
        print('Total rewards considered here: ', num_rewards)
        print('Total rewards not considered here: ', len(rewards_to_remove))
        print('Total assistant and manual rewards: ', len(assistant_reward_idx) + len(manual_reward_idx))

    return reward_idx