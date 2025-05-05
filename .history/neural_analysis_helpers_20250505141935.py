import numpy as np 
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import pandas as pd
import yaml
import seaborn as sns


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


def plot_avg_psth(average_psth, event='reward', zscoring=True, time_around=1, funcimg_frame_rate=45, save_psth=False, savepath='', filename=''):

    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = average_psth.shape[1]
    num_neurons = average_psth.shape[0]

    # Sort cells according to firing around event
    sortidx = np.argsort(np.argmax(average_psth, axis=1))

    data = average_psth.copy()
    if zscoring:
        # data = stats.zscore(data, axis=1)
        data = stats.zscore(data, axis=None)

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
        # sorting_data = stats.zscore(sorting_data, axis=1)
        # testing_data = stats.zscore(testing_data, axis=1)
        sorting_data = stats.zscore(sorting_data, axis=None)
        testing_data = stats.zscore(testing_data, axis=None)
    
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


def get_tuned_neurons(average_psth, event='reward', time_around=1, funcimg_frame_rate=45, plot_neurons=True):
    # Statistics to find neurons tuned to an event e.g. reward, lick, landmark entry etc.
    # TODO: bootstrapping / permutation test instead? 

    # Mann–Whitney U test comparing the period just before stimulus onset to the period directly after stimulus onset. 

    if isinstance(time_around, int):
        time_window = time_around * funcimg_frame_rate
    else:
        time_window = int(np.floor(time_around * funcimg_frame_rate))

    num_timebins = average_psth.shape[1]
    num_neurons = average_psth.shape[0]

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
            if time_around == int(time_around):
                xticklabels = [int(-time_around), 0, int(time_around)]
            else:
                xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
            ax.set_xticklabels(xticklabels)
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
    num_timebins = average_psth.shape[1]
    num_neurons = average_psth.shape[0]

    num_neurons = len(neurons)

    data = average_psth.copy()
    if zscoring:
        # data = stats.zscore(np.array(data), axis=1)
        data = stats.zscore(np.array(data), axis=None)

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
        # data = stats.zscore(data, axis=1)
        data = stats.zscore(data, axis=None)

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
            # data[goal] = stats.zscore(data[goal], axis=1)
            data[goal] = stats.zscore(data[goal], axis=None)

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
    '''Plot firing maps for all sessions and each goal, sorted by a specific goal.'''

    assert sorting_goal in all_average_psths[ref_session], 'This goal does not exist in the reference session.'
    time_window = time_around * funcimg_frame_rate # frames
    num_timebins = 2*time_window

    # Copy and optionally z-score
    data = []
    if isinstance(all_average_psths, list):
        for session in all_average_psths:
            session_data = {}
            for goal in session.keys():
                # session_data[goal] = stats.zscore(session[goal], axis=1) if zscoring else session[goal]
                session_data[goal] = stats.zscore(session[goal], axis=None) if zscoring else session[goal]
            data.append(session_data)

    elif isinstance(all_average_psths, dict):
        # Flatten the data
        for session_id, session in all_average_psths.items():  
            session_data = {}
            for goal in session.keys():
                # session_data[goal] = stats.zscore(session[goal], axis=1) if zscoring else session[goal]
                session_data[goal] = stats.zscore(session[goal], axis=None) if zscoring else session[goal]
            data.append(session_data)

    # Compute global vmin/vmax
    vmin = min([np.nanmin(session[goal]) for session in data for goal in session.keys()])
    vmax = max([np.nanmax(session[goal]) for session in data for goal in session.keys()])

    # Sort neurons consistently across sessions (using sorting_goal)
    sortidx = np.argsort(np.argmax(data[ref_session][sorting_goal], axis=1))  # reference the first session for sorting

    # Set up figure
    # Determine the number of sessions and max number of goals
    num_sessions = len(all_average_psths)
    goals_per_session = [sorted(data[s].keys()) for s in range(num_sessions)]
    max_goals = max(len(goals) for goals in goals_per_session)
    goal_label_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', '1': 'A', '2': 'B', '3': 'C', '4': 'D', 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}
    protocol_nums = sorted(set([cond.split()[0] for cond in conditions]))

    fig, ax = plt.subplots(num_sessions, max_goals, figsize=(3*max_goals, 3*num_sessions), sharex=True, sharey=True)

    if num_sessions == 1 or max_goals == 1:
        ax = np.atleast_2d(ax)
    ax = np.array(ax)

    # Plot
    for s in range(num_sessions):
        for g, goal in enumerate(goals_per_session[s]):
            print(goals_per_session[s])
            ax[s, g].imshow(data[s][goal][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
            ax[s, g].vlines(time_window-0.5, ymin=-0.5, ymax=data[s][goal].shape[0]-0.5, color='k', linewidth=0.5)
            ax[s, g].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
            if time_around == int(time_around):
                xticklabels = [int(-time_around), 0, int(time_around)]
            else:
                xticklabels = [round(-time_around, 1), 0, round(time_around, 1)]
            ax[s, g].set_xticklabels(xticklabels)
            ax[s, g].spines[['right', 'top']].set_visible(False)
            ax[s, g].set_title(goal_label_map.get(goal, str(goal)))
            if g == 0:
                ax[s, g].set_ylabel(f'{protocol_nums[s]}\nNeuron', labelpad=-5)
        ax[s,0].set_yticks([-0.5, data[ref_session][goals_per_session[0][0]].shape[0]-0.5])  
        ax[s,0].set_yticklabels([0, data[ref_session][goals_per_session[0][0]].shape[0]])

        # Hide unused axes in that row
        for g_unused in range(len(goals_per_session[s]), max_goals):
            ax[s, g_unused].axis('off')

    cbar = fig.colorbar(ax[0,0].images[0], ax=ax.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=0, fontsize=8)

    # fig, ax = plt.subplots(num_sessions, num_goals, figsize=(3*num_goals, 3*num_sessions), sharex=True, sharey=True)
    # if num_sessions == 1 or num_goals == 1:
    #     ax = np.atleast_2d(ax)
    # ax = np.array(ax)

    # for s in range(num_sessions):
    #     for g, goal in enumerate(sorted(data[s].keys())):
    #         ax[s,g].imshow(data[s][goal][sortidx, :], aspect='auto', vmin=vmin, vmax=vmax)
    #         ax[s,g].vlines(time_window-0.5, ymin=-0.5, ymax=data[s][goal].shape[0]-0.5, color='k', linewidth=0.5)
    #         ax[s,g].set_xticks([-0.5, num_timebins/2-0.5, num_timebins-0.5])
    #         ax[s,g].set_xticklabels([int(-time_around), 0, int(time_around)])
    #         ax[s,g].spines[['right', 'top']].set_visible(False)
    #         if s == 0:
    #             ax[s,g].set_title(goals[g])
    #         if g == 0:
    #             ax[s,g].set_ylabel(f'Session {s+1}\nNeuron')

    # ax[0,0].set_yticks([-0.5, data[s][goal].shape[0]-0.5])
    # ax[0,0].set_yticklabels([0, data[s][goal].shape[0]])
    # ax[0,0].set_ylabel('Neuron')

    # ax[1,0].set_yticks([-0.5, data[s][goal].shape[0]-0.5])
    # ax[1,0].set_yticklabels([0, data[s][goal].shape[0]])
    # ax[1,0].set_ylabel('Neuron')

    # # Add colorbar
    # cbar = fig.colorbar(ax[0,0].images[0], ax=ax.ravel().tolist(), shrink=0.6)
    # cbar.set_ticks([vmin, vmax])
    # cbar.ax.set_yticklabels([str(int(round(vmin))), str(int(round(vmax)))], fontsize=8)
    # cbar.set_label(r'z-scored $\Delta$F/F0' if zscoring else r'$\Delta$F/F0', rotation=270, labelpad=10, fontsize=8)

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
            # data[i] = stats.zscore(data[i], axis=1)
            data[i] = stats.zscore(data[i], axis=None)

    # Find global vmin and vmax across all conditions
    vmin = min([np.nanmin(d) for d in data if d.size > 0])
    vmax = max([np.nanmax(d) for d in data if d.size > 0])

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
        

def get_map_correlation(psths, average_psths, conditions, zscoring=True, reference=0, color_scheme=None, save_plot=False, savepath='', savedir='', filename=''):
    '''
    Get the firing map correlation among different conditions against a reference. 
    The correlation for the reference is calculated by randomly selecting half the trials.
    NOTE: The reference is the index of the data if the data are a list, or a nested dict (will get flattened into a list), but it is a key of the data if the data are a dict. 
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
                average_psth_data.append(stats.zscore(np.array(average_psths[c]), axis=None))
            # psth_data = stats.zscore(np.array(psth_data), axis=2)
                psth_data.append(stats.zscore(np.array(psths[c]), axis=None))
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
                        # d = stats.zscore(d, axis=1)  
                        d = stats.zscore(d, axis=None)  
                        # ref = stats.zscore(ref, axis=1)
                        ref = stats.zscore(ref, axis=None)
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
                    # average_psth_data[i] = stats.zscore(average_psth_data[i], axis=1)
                    average_psth_data[i] = stats.zscore(average_psth_data[i], axis=None)
                    # psth_data[i] = stats.zscore(psth_data[i], axis=1)
                    psth_data[i] = stats.zscore(psth_data[i], axis=None)

            data_indices = list(average_psth_data.keys())
            if reference not in average_psth_data.keys():
                raise ValueError(f'Reference condition {reference} should be one of the keys of the input dict.')
            ref_cond = data_indices.index(reference)

    num_neurons = average_psth_data[reference].shape[0]
    
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
        for n in range(num_neurons):

            if idx == reference:
                if np.all(np.isfinite(sorting_data[n])) and np.all(np.isfinite(sorting_data[n])):
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
    labels = []
    for i, cond in enumerate(conditions):
        if isinstance(average_psths, list):
            labels.append(f"{cond}\nvs\n{conditions[ref_cond]}")
        elif isinstance(average_psths, dict):
            labels.append(f"{cond} vs {conditions[ref_cond]}")

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

    # Fallback color scheme if none is given
    if color_scheme is None:
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


def get_map_correlation_matrix(all_average_psths, conditions, zscoring=True, save_plot=False, savepath='', savedir='', filename=''):
    '''
    Calculate pairwise PSTH correlation across all sessions and goals.
    '''
    num_sessions = len(all_average_psths)
    num_goals = len(all_average_psths[0])

    # Flatten all data: [(session 0 goal A), (session 0 goal B), ..., (session 1 goal A), ...]
    data = []
    for s in range(num_sessions):
        for goal in all_average_psths[s].keys():  
            d = all_average_psths[s][goal]
            if zscoring:
                # d = stats.zscore(d, axis=1)  # z-score along time
                d = stats.zscore(d, axis=None)
            data.append(d)

    num_conditions = len(data)  
    assert num_conditions == len(data), 'The length of the input data does not match the number of conditions.'

    # Initialize correlation matrix
    correlation_matrix = np.zeros((num_conditions, num_conditions))

    # Calculate correlations
    for i in range(num_conditions):
        for j in range(num_conditions):
            correlations = []
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
    im = ax.imshow(correlation_matrix, cmap='viridis', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean neuron correlation', fontsize=10)

    # Axis labels
    ax.set_xticks(np.arange(num_conditions))
    ax.set_yticks(np.arange(num_conditions))
    ax.set_xticklabels(conditions, rotation=90)
    ax.set_yticklabels(conditions)
    ax.set_title('All Sessions and Goals PSTH Correlation')

    plt.tight_layout()

    if save_plot:
        output_path = os.path.join(savepath, savedir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, f'{filename}.png'))

    plt.show()

    return correlation_matrix


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

        for i, (lm_start, lm_end) in enumerate(session['all_landmarks']):
            idx_candidates = np.where((positions[search_start:] >= lm_start) & (positions[search_start:] <= lm_end))[0]
            if len(idx_candidates) > 0:
                lm_entry_idx.append(idx_candidates[0] + search_start)
                lm_exit_idx.append(idx_candidates[-1] + search_start)  # TODO: confirm
                search_start += idx_candidates[0] 
            else:
                print(f"Warning: no match found for landmark {i} with bounds {lm_start}-{lm_end}")
                lm_entry_idx.append(None)

    else:
        for lm_start in session['all_landmarks'][:,0]:
            lm_entry_idx.append(np.where(positions >= lm_start)[0][0])
            # lm_entry_idx2.append(int(np.argmin(np.abs(positions - lm_start)))) 

        for lm_end in session['all_landmarks'][:,1]:
            # lm_exit_idx.append(int(np.argmin(np.abs(positions - lm_end))))
            lm_exit_idx.append(np.where(positions <= lm_end)[0][-1])

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
    '''Define which landmarks belong to goals, non-goals and test.'''

    if num_landmarks == 10:     # T5 and T6
        if sequence == 'ABAB':
            goal_landmark_id = [1, 3, 5, 7]
            test_landmark_id = 9
        elif sequence == 'AABB':  
            goal_landmark_id = [0, 1, 4, 5]
            test_landmark_id = 8
        non_goal_landmark_id = np.setxor1d(np.arange(0,num_landmarks), np.append(goal_landmark_id, test_landmark_id))
 
    elif num_landmarks == 2:    # T3 and T4
        lms = np.unique(session['all_lms'])
        goal_landmark_id = session['all_lms'][session['goal_idx'][0]]
        non_goal_landmark_id = np.setdiff1d(lms, goal_landmark_id)[0]
        test_landmark_id = None

    # Get the landmarks that belong to each condition  
    goals_idx = np.where(np.isin(session['all_lms'], goal_landmark_id))[0]
    non_goals_idx = np.where(np.isin(session['all_lms'], non_goal_landmark_id))[0]
    test_idx = np.where(np.isin(session['all_lms'], test_landmark_id))[0] if test_landmark_id is None else None
    
    return goals_idx, non_goals_idx, test_idx


def get_landmark_category_rew_idx(sequence, num_landmarks, session, VR_data, nidaq_data):
    '''Find indices also in non-goal landmarks corresponding to the same time after landmark entry as mean reward time lag.'''
    
    reward_idx = get_rewards(VR_data, nidaq_data, print_output=True)

    rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx = get_landmark_category_entries(VR_data, nidaq_data, sequence, num_landmarks, session)
    
    # Calculate time lag between landmark entry and reward delivery
    rew_time_lag = np.round(np.mean(reward_idx - rew_lm_entry_idx))
    print('Reward time lag from lm entry: ', rew_time_lag)

    # Find where reward would be on average if these landmarks were rewarded
    miss_rew_idx = miss_lm_entry_idx + rew_time_lag
    nongoal_rew_idx = nongoal_lm_entry_idx + rew_time_lag  
    test_rew_idx = test_lm_entry_idx + rew_time_lag

    return rew_time_lag, reward_idx, miss_rew_idx, nongoal_rew_idx, test_rew_idx


def get_landmark_category_entries(VR_data, nidaq_data, sequence, num_landmarks, session):
    '''Find the indices of landmark entry for different types of landmarks: rewarded, miss, non-goal, test.'''
    
    lm_entry_idx, _ = get_lm_entry_exit(session, positions=nidaq_data['position'])

    # Find category for each landmark 
    goals_idx, non_goals_idx, test_idx = get_landmark_categories(sequence, num_landmarks, session)

    # Find the rewarded landmarks 
    rewarded_landmarks = get_rewarded_landmarks(VR_data, nidaq_data, session)

    # Find landmark entry indices for each landmark category
    rew_lm_entry_idx = [lm_entry_idx[i] for i in rewarded_landmarks]
    miss_lm_entry_idx = np.array([lm_entry_idx[i] for i in goals_idx if i not in rewarded_landmarks])
    nongoal_lm_entry_idx = np.array([lm_entry_idx[i] for i in non_goals_idx])
    test_lm_entry_idx = np.array([lm_entry_idx[i] for i in test_idx])

    assert len(rew_lm_entry_idx) + len(miss_lm_entry_idx) + len(nongoal_lm_entry_idx) + len(test_lm_entry_idx) == len(session['all_lms']), 'Some landmarks have not been considered.'

    return rew_lm_entry_idx, miss_lm_entry_idx, nongoal_lm_entry_idx, test_lm_entry_idx


def get_rewarded_landmarks(VR_data, nidaq_data, session):
    '''Find the indices of rewarded (lick-triggered) landmarks.'''

    reward_idx = get_rewards(VR_data, nidaq_data, print_output=False)

    # Find rewarded landmarks 
    reward_positions = nidaq_data['position'][reward_idx]

    rewarded_landmarks = [i for i, (start, end) in enumerate(session['all_landmarks']) 
                        if np.any((reward_positions >= start) & (reward_positions <= end))]  # TODO: what is wrong with the last reward? 
    
    return rewarded_landmarks


def get_rewards(VR_data, nidaq_data, print_output=False):
    '''Find the indices of rewards in the nidaq logging file.'''

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
    reward_idx = reward_idx[0:-1]  # TODO: Deal with last reward...
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