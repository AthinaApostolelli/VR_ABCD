import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import friedmanchisquare, wilcoxon, norm, kruskal, mannwhitneyu
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def get_lm_data(session, neurons, patches, goal_patches, non_goal_patches, time_around,
                lm_entry_idx, lm_exit_idx, dF, condition='next', n_bins=31, funcimg_frame_rate=45,
                zscoring=False, plot=True):
    """
    Extract dF/F aligned to a chosen landmark for each patch, supporting multiple neurons.
    
    Parameters
    ----------
    session : dict
        Session dictionary with 'reward_idx', 'miss_rew_idx', 'nongoal_rew_idx'.
    neurons : dict or list
        Indexed by session; contains neuron IDs.
    patches : list of arrays
        Patch trial indices.
    goal_patches, non_goal_patches : list of arrays
        Patches categorized by goal or non-goal.
    lm_entry_idx, lm_exit_idx : array-like
        Entry and exit indices for landmarks.
    dF : np.ndarray
        Fluorescence data, shape (n_neurons, n_timepoints).
    condition : str
        Landmark to extract neural data for
        - next: first after patch end
        - last: last in patch 
        - prev: second to last in patch
    n_bins : int
        Total number of bins (split equally pre/post).
    plot : bool
        Whether to plot or not. 
    
    Returns:
        - Dictionary with goal/non-goal windows and patches_by_length
        - Matplotlib axes for goal and non-goal plots
    """
    event_idx = np.sort(np.concatenate([session['reward_idx'], session['miss_rew_idx'], session['nongoal_rew_idx']])).astype(int)

    neurons = np.atleast_1d(neurons)
    n_neurons = len(neurons)
    
    # Slice dF for the neurons we are analyzing
    dF_sel = dF[neurons, :]
    
    # Flatten patches
    goal_patches_flat = np.unique(np.concatenate([np.ravel(p) for p in goal_patches]))
    non_goal_patches_flat = np.unique(np.concatenate([np.ravel(p) for p in non_goal_patches]))
    
    goal_patches_by_length = {}
    non_goal_patches_by_length = {}

    # Labels
    if condition == 'next':
        label_goal, label_non_goal = 'B test', 'A test'
    elif condition == 'last':
        label_goal, label_non_goal = 'B', 'A'
    elif condition == 'prev':
        label_goal, label_non_goal = 'A', 'B'
    else:
        raise ValueError("condition must be 'prev', 'last', or 'next'")
    
    # Loop over patches
    for p, patch in enumerate(patches):
        patch_len = len(patch)
        
        # Determine landmark index
        if condition == 'next':
            lm = patch[-1] + 1
        elif condition == 'last':
            lm = patch[-1]
        else:  # prev
            lm = patch[-1] - 1
        
        # Skip invalid
        if lm < 0 or lm >= len(lm_entry_idx):
            continue
        
        start, end = lm_entry_idx[lm], lm_exit_idx[lm]
        r = event_idx[lm]
        if end == r:
            continue
        
        # Skip missed goal landmarks
        if np.isin(r, session['miss_rew_idx']): 
            # print('missed')
            continue
         
        if isinstance(time_around, (int, float)):
            start_time = -time_around
            end_time = time_around
        elif isinstance(time_around, (tuple, list)) and len(time_around) == 2:
            start_time, end_time = time_around
        start_frames = int(np.floor(start_time * funcimg_frame_rate))
        end_frames = int(np.ceil(end_time * funcimg_frame_rate))

        # Get indices for each event
        window = np.arange(start_frames, end_frames)
        window_indices = np.add.outer(r, window).astype(int)

        # Remove last events if close to session end 
        valid_mask = window_indices[-1] < dF.shape[1]
        valid_window_indices = window_indices[valid_mask]
        
        # Create binned array (n_neurons x n_bins)
        binned = np.full((n_neurons, len(window)), np.nan)
        binned[:,:] = dF_sel[:, np.squeeze(valid_window_indices)]
        
        # z-score data 
        if zscoring:
            binned = stats.zscore(np.array(binned), axis=1)

        # Append to goal or non-goal dictionary
        if patch[-1] in goal_patches_flat:
            goal_patches_by_length.setdefault(patch_len, []).append(binned)
        elif patch[-1] in non_goal_patches_flat:
            non_goal_patches_by_length.setdefault(patch_len, []).append(binned)
    
    # Stack lists into arrays: shape (n_patches, n_neurons, n_bins)
    for length in goal_patches_by_length:
        goal_patches_by_length[length] = np.stack(goal_patches_by_length[length], axis=0)
    for length in non_goal_patches_by_length:
        non_goal_patches_by_length[length] = np.stack(non_goal_patches_by_length[length], axis=0)
    
    if plot:
        # --- Plot Goal Patches ---
        fig_goal, goal_ax = plt.subplots(1, len(goal_patches_by_length), figsize=(3*len(goal_patches_by_length), 3), sharey=True, squeeze=False)
        for i, (length, arr) in enumerate(sorted(goal_patches_by_length.items())):
            if n_neurons == 1:
                mean_data = np.nanmean(np.squeeze(arr, axis=1), axis=0)
            else:
                mean_data = np.nanmean(arr, axis=(0,1))
            goal_ax[0,i].plot(mean_data, color='blue', label=label_goal)
            goal_ax[0,i].axvline(x=n_bins // 2, color='gray', linestyle='--')
            goal_ax[0,i].set_title(f'Goal: Patch length {length}')
            goal_ax[0,i].set_xlabel('Normalized time')
            if i == 0:
                goal_ax[0,i].set_ylabel('dF/F')
        plt.tight_layout()
    
        # --- Plot Non-Goal Patches ---
        fig_non_goal, non_goal_ax = plt.subplots(1, len(non_goal_patches_by_length), figsize=(3*len(non_goal_patches_by_length), 3), sharey=True, squeeze=False)
        for i, (length, arr) in enumerate(sorted(non_goal_patches_by_length.items())):
            if n_neurons == 1:
                mean_data = np.nanmean(np.squeeze(arr, axis=1), axis=0)
            else:
                mean_data = np.nanmean(arr, axis=(0,1))
            non_goal_ax[0,i].plot(mean_data, color='orange', label=label_non_goal)
            non_goal_ax[0,i].axvline(x=n_bins // 2, color='gray', linestyle='--')
            non_goal_ax[0,i].set_title(f'Non-Goal: Patch length {length}')
            non_goal_ax[0,i].set_xlabel('Normalized time')
            if i == 0:
                non_goal_ax[0,i].set_ylabel('dF/F')
        plt.tight_layout()
    
        return {
            "goal_patches_by_length": goal_patches_by_length,
            "non_goal_patches_by_length": non_goal_patches_by_length
        }, goal_ax, non_goal_ax
    
    else:
        return {
            "goal_patches_by_length": goal_patches_by_length,
            "non_goal_patches_by_length": non_goal_patches_by_length
        }, None, None


def compare_lms_in_AB_patches(neurons, session, patches, goal_patches, non_goal_patches, lm_entry_idx, 
                              lm_exit_idx, dF, time_around, n_bins=10, zscoring=False, plot=True, plot_neurons=None):
    
    next_lm_data, _, _ = get_lm_data(session, neurons, patches, goal_patches, non_goal_patches, time_around,
                lm_entry_idx, lm_exit_idx, dF, condition='next', n_bins=n_bins, zscoring=zscoring, plot=False)

    lm_data, _, _ = get_lm_data(session, neurons, patches, goal_patches, non_goal_patches, time_around,
                    lm_entry_idx, lm_exit_idx, dF, condition='last', n_bins=n_bins, zscoring=zscoring, plot=False)

    prev_lm_data, _, _ = get_lm_data(session, neurons, patches, goal_patches, non_goal_patches, time_around,
                    lm_entry_idx, lm_exit_idx, dF, condition='prev', n_bins=n_bins, zscoring=zscoring, plot=False)

    # Extract goal/non-goal by length for all three conditions
    goal_data = {'prev': prev_lm_data['goal_patches_by_length'],
                'last': lm_data['goal_patches_by_length'],
                'next': next_lm_data['goal_patches_by_length']}

    non_goal_data = {'prev': prev_lm_data['non_goal_patches_by_length'],
                    'last': lm_data['non_goal_patches_by_length'],
                    'next': next_lm_data['non_goal_patches_by_length']}

    line_styles = {'prev': ':', 'last': '-', 'next': '--'}
    colors = {'goal': 'blue', 'non_goal': 'orange'}

    if plot:
        # --- Plot Goal Patches ---
        lengths = sorted(goal_data['last'].keys())
        fig, goal_ax = plt.subplots(1, len(lengths), figsize=(4*len(lengths), 3), sharey=True)
        if len(lengths) == 1:
            goal_ax = [goal_ax]

        for i, length in enumerate(lengths):
            ax = goal_ax[i]
            for id, cond in zip([' A', ' B', ' B-test'], ['prev', 'last', 'next']):
                if length not in goal_data[cond]:
                    continue 
                patches_array = goal_data[cond][length]  # shape: n_patches x n_bins
                if patches_array.shape[1] == 1:
                    mean_data = np.nanmean(np.squeeze(patches_array, axis=1), axis=0)
                else:
                    mean_data = np.nanmean(patches_array, axis=(0,1))
                ax.plot(mean_data, color=colors['goal'], linestyle=line_styles[cond], label=[cond+id])
                ax.axvline(x=mean_data.shape[-1] // 2, color='gray', linestyle='--')
            ax.set_title(f'Goal: Patch length {length}')
            ax.set_xlabel('Normalized time')
            if i == 0:
                ax.set_ylabel('dF/F')
            ax.legend()
        plt.tight_layout()

        # --- Plot Non-Goal Patches ---
        lengths = sorted(non_goal_data['last'].keys())
        fig, non_goal_ax = plt.subplots(1, len(lengths), figsize=(4*len(lengths), 3), sharey=True)
        if len(lengths) == 1:
            non_goal_ax = [non_goal_ax]

        for i, length in enumerate(lengths):
            ax = non_goal_ax[i]
            for id, cond in zip([' B', ' A', ' A-test'], ['prev', 'last', 'next']):
                if length not in non_goal_data[cond]:
                    continue 
                patches_array = non_goal_data[cond][length]
                if patches_array.shape[1] == 1:
                    mean_data = np.nanmean(np.squeeze(patches_array, axis=1), axis=0)
                else:
                    mean_data = np.nanmean(patches_array, axis=(0,1))
                ax.plot(mean_data, color=colors['non_goal'], linestyle=line_styles[cond], label=[cond+id])
                ax.axvline(x=mean_data.shape[-1] // 2, color='gray', linestyle='--')
            ax.set_title(f'Non-Goal: Patch length {length}')
            ax.set_xlabel('Normalized time')
            if i == 0:
                ax.set_ylabel('dF/F')
            ax.legend()
        plt.tight_layout()

    if plot_neurons is not None:
        # --- Plot Goal Patches ---
        lengths = sorted(goal_data['last'].keys())
        for n, neuron in enumerate(neurons[:plot_neurons]):
            _, ax = plt.subplots(1, len(lengths), figsize=(4*len(lengths), 3), sharey=False)
            ax = ax.ravel()
            for i, length in enumerate(lengths):
                for id, cond in zip([' A', ' B', ' B-test'], ['prev', 'last', 'next']):
                    if length not in goal_data[cond]:
                        continue 
                    patches_array = goal_data[cond][length]  # shape: n_patches x n_bins
                    mean_data = np.nanmean(patches_array[:,n,:], axis=0)
                    ax[i].plot(mean_data, color=colors['goal'], linestyle=line_styles[cond], label=[cond+id])
                    ax[i].axvline(x=mean_data.shape[-1] // 2, color='gray', linestyle='--')
                ax[i].set_title(f'Goal: Patch length {length}')
                ax[i].set_xlabel('Normalized time')
                if i == 0:
                    ax[i].set_ylabel('dF/F')
                ax[i].legend()
            plt.suptitle(f'Neuron {n}: {neuron}')
            plt.tight_layout()

        # --- Plot Non-Goal Patches ---
        lengths = sorted(non_goal_data['last'].keys())
        for n, neuron in enumerate(neurons[:plot_neurons]):
            _, ax = plt.subplots(1, len(lengths), figsize=(4*len(lengths), 3), sharey=False)
            ax = ax.ravel()
            for i, length in enumerate(lengths):
                for id, cond in zip([' B', ' A', ' A-test'], ['prev', 'last', 'next']):
                    if length not in non_goal_data[cond]:
                        continue 
                    patches_array = non_goal_data[cond][length]  # shape: n_patches x n_bins
                    mean_data = np.nanmean(patches_array[:,n,:], axis=0)
                    ax[i].plot(mean_data, color=colors['non_goal'], linestyle=line_styles[cond], label=[cond+id])
                    ax[i].axvline(x=mean_data.shape[-1] // 2, color='gray', linestyle='--')
                ax[i].set_title(f'Non-Goal: Patch length {length}')
                ax[i].set_xlabel('Normalized time')
                if i == 0:
                    ax[i].set_ylabel('dF/F')
                ax[i].legend()
            plt.suptitle(f'Neuron {n}: {neuron}')
            plt.tight_layout()

    return next_lm_data, lm_data, prev_lm_data


def compute_lm_mean(next_lm_data, lm_data, prev_lm_data):
    # Collapse all bins into per-trial means for each condition and goal type

    all_data = {
        'next': next_lm_data,
        'last': lm_data,
        'prev': prev_lm_data
    }

    goal_means = {}
    non_goal_means = {}

    for cond, data in all_data.items():
        goal_means[cond] = {}
        non_goal_means[cond] = {}

        # Goal patches
        for length, arr in data['goal_patches_by_length'].items():
            # arr shape: (n_patches, n_neurons, n_bins)
            goal_means[cond][length] = np.nanmean(arr, axis=(0,2))  # → (n_neurons,)
        # Non-goal patches
        for length, arr in data['non_goal_patches_by_length'].items():
            non_goal_means[cond][length] = np.nanmean(arr, axis=(0,2))

    return goal_means, non_goal_means


def compute_neuron_lm_mean(next_lm_data, lm_data, prev_lm_data):
    all_data = {
        'next': next_lm_data,
        'last': lm_data,
        'prev': prev_lm_data
    }

    neuron_goal_means = {}
    neuron_non_goal_means = {}

    for cond, data in all_data.items():

        # collect all length arrays
        goal_arrays = list(data['goal_patches_by_length'].values())
        non_goal_arrays = list(data['non_goal_patches_by_length'].values())

        # result containers
        goal_list = []
        non_goal_list = []

        # --- GOAL ---
        for arr in goal_arrays:
            # arr shape: (n_patches, n_neurons, n_bins)
            # mean over bins → (n_patches, n_neurons)
            patch_means = np.nanmean(arr, axis=2)
            goal_list.append(patch_means)

        # concatenate across lengths
        neuron_goal_means[cond] = np.concatenate(goal_list, axis=0) if goal_list else None

        # --- NON-GOAL ---
        for arr in non_goal_arrays:
            patch_means = np.nanmean(arr, axis=2)
            non_goal_list.append(patch_means)

        neuron_non_goal_means[cond] = np.concatenate(non_goal_list, axis=0) if non_goal_list else None

    return neuron_goal_means, neuron_non_goal_means


def get_responsive_neurons(psth, event='reward', time_around=1, funcimg_frame_rate=45, plot_neurons=True):
    """
    Mann-Whitney U test comparing firing before and after event.
    
    Parameters
    ----------
    psth : array, shape (neurons, trials, timebins)
    event : str
        Event name
    time_around : int/float or tuple
        If int/float: window in seconds, symmetric around 0 (e.g. 1 → -1 to +1).
        If tuple: (start, end) in seconds (e.g. (-1, 2)).
    funcimg_frame_rate : int
        Imaging frame rate (Hz).
    plot_neurons : bool
        Whether to plot significant neurons.
    """
    # TODO: bootstrapping / permutation test instead? 

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
    num_neurons = psth.shape[0]

    event_frame = -start_frames  # event aligned at t=0
    before_idx = slice(0, event_frame)          # frames before event
    after_idx  = slice(event_frame, event_frame + end_frames)  # frames after event

    # Average across timebins for each trial
    before_event_firing = np.mean(psth[:, :, before_idx], axis=2)
    after_event_firing  = np.mean(psth[:, :, after_idx], axis=2)
    # print(before_event_firing.shape, after_event_firing.shape)

    # Perform the test using all trials for each neuron
    wilcoxon_stat = np.zeros((num_neurons, 1))
    wilcoxon_pval = np.zeros((num_neurons, 1))
    for n in range(num_neurons):
        wilcoxon_stat[n], wilcoxon_pval[n] = stats.wilcoxon(before_event_firing[n, :], after_event_firing[n, :]) #, method=stats.PermutationMethod(n_resamples=1000))

    # Criteria to define tuned neurons
    # 1. p-value
    criterion1 = np.where(wilcoxon_pval < 0.01)[0]   

    # 2. peak in the 1s after event > mean + 2*std of the 1s before the event
    average_psth = np.mean(psth, axis=1)
    before_event_avg_firing = average_psth[:, before_idx]
    after_event_avg_firing = average_psth[:, after_idx]
    criterion2_high = np.where(np.max(after_event_avg_firing, axis=1) > (np.mean(before_event_avg_firing, axis=1) + 2 * np.std(before_event_avg_firing, axis=1)))[0]
    criterion2_low = np.where(np.max(after_event_avg_firing, axis=1) < (np.mean(before_event_avg_firing, axis=1)))[0] #- 2 * np.std(before_event_avg_firing, axis=1)))[0]
    
    tuned_neurons = criterion1
    tuned_neurons_high = np.intersect1d(criterion1, criterion2_high)
    tuned_neurons_low = np.setdiff1d(tuned_neurons, tuned_neurons_high)
    # tuned_neurons = np.intersect1d(criterion1, criterion2)
    print(f'{len(tuned_neurons)} neurons are tuned to {event}.')

    # Plot firing for a few significant neurons
    if plot_neurons:
        for n in tuned_neurons[0:20]:
            fig, ax = plt.subplots(1, 1, figsize=(2,2), sharey=True)
            ax.plot(average_psth[n, :])      
            event_frame = -start_frames  
            # ax.plot(average_psth[n, :])      
            ax.axvspan(event_frame, event_frame + end_frames, color='gray', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_xticks([event_frame + start_frames, event_frame, event_frame + end_frames])
            ax.set_xticklabels([start_time, 0, end_time])
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_ylabel('DF/F')
            ax.set_title(f'Neuron {n}, p-value {wilcoxon_pval[n]}')

    return tuned_neurons, tuned_neurons_high, tuned_neurons_low, wilcoxon_stat, wilcoxon_pval


def temporal_bin_ABB_firing(ABB_patches, ABB_patches_idx, cell, dF, bins=90, plot=True):
    binned_phase_firing = np.zeros((len(ABB_patches), bins))

    for i in range(len(ABB_patches)):
        phase_frames = np.arange(ABB_patches_idx[i][0], ABB_patches_idx[i][1])
        bin_edges = np.linspace(ABB_patches_idx[i][0], ABB_patches_idx[i][1], bins+1)
        phase_firing = dF[cell, phase_frames]
        
        bin_ix = np.digitize(phase_frames, bin_edges)
        for j in range(bins):
            binned_phase_firing[i,j] = np.mean(phase_firing[bin_ix == j+1])

    if plot:
        fig = plt.figure(figsize=(3,3))
        ax1 = fig.add_subplot(111)
        cax = ax1.imshow(binned_phase_firing, aspect='auto', cmap='viridis', interpolation='none')
        ax1.set_title(f'Cell {cell} - Binned Firing Rates')
        plt.colorbar(cax, ax=ax1, label='dF/F')
        plt.tight_layout()

    return binned_phase_firing


def spatial_bin_ABB_firing(ABB_patches, ABB_patches_idx, cell, dF, session, bins=90, plot=True):
    positions = session['position']
    binned_phase_firing = np.zeros((len(ABB_patches), bins))

    for i in range(len(ABB_patches)):
        phase_frames = np.arange(ABB_patches_idx[i][0], ABB_patches_idx[i][1])
        phase_positions = positions[phase_frames]
        bin_edges = np.linspace(phase_positions.min(), phase_positions.max(), bins+1)
        phase_firing = dF[cell, phase_frames]

        bin_ix = np.digitize(phase_positions, bin_edges)
        for j in range(bins):
            binned_phase_firing[i,j] = np.mean(phase_firing[bin_ix == j+1])

    if plot:
        fig = plt.figure(figsize=(3,3))
        ax1 = fig.add_subplot(111)
        cax = ax1.imshow(binned_phase_firing, aspect='auto', cmap='viridis', interpolation='none')
        ax1.set_title(f'Cell {cell} - Binned Firing Rates')
        plt.colorbar(cax, ax=ax1, label='dF/F')
        plt.tight_layout()

    return binned_phase_firing


def get_spatial_and_temporal_ABB_binning(ABB_patches, ABB_patches_idx, neurons, dF, session, bins=90):
    
    temporal_ABB_firing = {}
    spatial_ABB_firing = {}
    for cell in neurons:
        temporal_ABB_firing[cell] = temporal_bin_ABB_firing(ABB_patches, ABB_patches_idx, cell, dF, bins, plot=False)
        spatial_ABB_firing[cell] = spatial_bin_ABB_firing(ABB_patches, ABB_patches_idx, cell, dF, session, bins, plot=False)

    # Get the mean across patches for all neurons
    avg_temporal_ABB_firing = np.empty((len(neurons), bins))
    avg_spatial_ABB_firing = np.empty((len(neurons), bins))
    for n, cell in enumerate(neurons):
        avg_temporal_ABB_firing[n] = np.nanmean(temporal_ABB_firing[cell], axis=0)
        avg_spatial_ABB_firing[n] = np.nanmean(spatial_ABB_firing[cell], axis=0)

    # Z-score per neuron
    zscored_avg_temporal_ABB_firing = stats.zscore(avg_temporal_ABB_firing, axis=1)
    zscored_avg_spatial_ABB_firing = stats.zscore(avg_spatial_ABB_firing, axis=1)

    # Sort according to max firing 
    peak_bins = np.argmax(zscored_avg_temporal_ABB_firing, axis=1)
    sort_order = np.argsort(peak_bins)
    zscored_sorted_temporal = zscored_avg_temporal_ABB_firing[sort_order]

    peak_bins = np.argmax(zscored_avg_spatial_ABB_firing, axis=1)
    sort_order = np.argsort(peak_bins)
    zscored_sorted_spatial = zscored_avg_spatial_ABB_firing[sort_order]

    # Plot
    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(121)
    cax1 = ax1.imshow(zscored_sorted_temporal, aspect='auto', cmap='viridis', interpolation='none')
    ax1.set_title(f'Binned Firing Rates (Temporal)')
    ax1.set_xlabel('Time bins')
    cb1 = fig.colorbar(cax1, ax=ax1, label='dF/F')  

    ax2 = fig.add_subplot(122)
    cax2 = ax2.imshow(zscored_sorted_spatial, aspect='auto', cmap='viridis', interpolation='none')
    ax2.set_title(f'Binned Firing Rates (Spatial)')
    ax2.set_xlabel('Position bins')
    cb2 = fig.colorbar(cax2, ax=ax2, label='dF/F')  

    for ax in [ax1, ax2]:
        ax.set_yticks([0, len(neurons)-1])
        ax.set_yticklabels([0, len(neurons)])
        ax.set_xticks([0, bins-1])
        ax.set_xticklabels([0, bins])
        ax.set_ylabel('Neurons', labelpad=-5)
        
    plt.tight_layout()

    # Collect all data into a dict
    binned_ABB_firing_rates = {}
    binned_ABB_firing_rates['temporal_ABB_firing'] = temporal_ABB_firing
    binned_ABB_firing_rates['spatial_ABB_firing'] = spatial_ABB_firing
    binned_ABB_firing_rates['avg_temporal_ABB_firing'] = avg_temporal_ABB_firing
    binned_ABB_firing_rates['avg_spatial_ABB_firing'] = avg_spatial_ABB_firing
    binned_ABB_firing_rates['zscored_sorted_temporal'] = zscored_sorted_temporal
    binned_ABB_firing_rates['zscored_sorted_spatial'] = zscored_sorted_spatial

    return binned_ABB_firing_rates


def find_cells_with_ABB_peaks(neurons, binned_activity, condition='temporal', plot=True):
    # ABB_patch_length_cm = session4['position'][ABB_patches_idx[0][1]] - session4['position'][ABB_patches_idx[0][0]]
    # bin_size_cm = ABB_patch_length_cm / nbins
    # place_field_size_cm = 10    # ~ half a lm 
    # place_field_bins = np.ceil(place_field_size_cm / bin_size_cm)

    peak_cells = []
    for n, cell in enumerate(neurons):
        if condition == 'temporal':
            smoothed = gaussian_filter1d(binned_activity['avg_temporal_ABB_firing'][n], sigma=2, mode='nearest')
        elif condition == 'spatial':
            smoothed = gaussian_filter1d(binned_activity['avg_spatial_ABB_firing'][n], sigma=2, mode='nearest')
        data = smoothed
        
        baseline = np.median(data)
        height = 1.2 * baseline

        peaks, props = find_peaks(data, distance=20, height=height)
        edge_peaks = []
        if data[0] > data[1] and (height is None or data[0] >= height):
            edge_peaks.append(0)
        if data[-1] > data[-2] and (height is None or data[-1] >= height):
            edge_peaks.append(len(data) - 1)
        all_peaks = np.sort(np.concatenate([peaks, edge_peaks])).astype(int)

        if len(peaks) > 0:
            peak_cells.append(cell)

        if plot:
            plt.figure(figsize=(4,3))
            plt.plot(data)
            plt.hlines(baseline, xmin=0, xmax=len(data), colors='k', linestyles='--')
            plt.hlines(height, xmin=0, xmax=len(data), colors='g', linestyles='--')
            plt.scatter(all_peaks, data[all_peaks], color='r')
            plt.title(f'Neuron {cell}')

    return peak_cells


# --------- STATISTICS --------- #
def kendalls_W(chi2, N, k):
    """Effect size for Friedman test."""
    return chi2 / (N * (k - 1))

def rank_biserial_from_wilcoxon(z, N):
    """Rank-biserial correlation effect size for Wilcoxon."""
    return z / np.sqrt(N)

def compute_population_stats(goal_means, conditions, idx):
    stats = {}

    # Loop over available patch lengths
    all_lengths = sorted(set().union(*[goal_means[c].keys() for c in conditions]))

    for length in all_lengths:

        # Make sure all conditions contain this patch length
        if not all(length in goal_means[c] for c in conditions):
            print(f'Skipping {length}-length patches, because data for one or more conditions are missing.')
            continue

        # Build matrix (n_neurons, 3)
        M = np.vstack([
            goal_means['prev'][length],
            goal_means['last'][length],
            goal_means['next'][length]
        ]).T

        # # Remove neurons with NaNs
        # good = ~np.isnan(M).any(axis=1)
        # M = M[good]

        if M.shape[0] < 3:    # need at least 3 neurons for Friedman
            continue

        n_neurons = M.shape[0]

        # ----------------------------
        # 1. Friedman (omnibus test)
        # ----------------------------
        chi2, p_friedman = friedmanchisquare(M[:,0], M[:,1], M[:,2])

        # effect size: Kendall's W
        W = kendalls_W(chi2, N=n_neurons, k=3)

        # ----------------------------
        # 2. Pairwise Wilcoxon tests
        # ----------------------------
        pairwise = {}
        pairs = [('prev','last'), ('last','next'), ('prev','next')]

        for a, b in pairs:
            # Wilcoxon returns a statistic but not Z, so compute Z manually
            stat, p_w = wilcoxon(M[:, idx[a]], M[:, idx[b]])
            
            # Compute Z-score from p-value (two-sided)
            # Wilcoxon in scipy uses two-sided p-values by default.
            if p_w == 0:
                # Avoid inf z-score
                z = np.sign(stat - (n_neurons*(n_neurons+1)/4)) * 8.0  
            else:
                z = norm.ppf(p_w / 2) * -1  # two-sided → divide by 2

            r_rb = rank_biserial_from_wilcoxon(z, n_neurons)

            pairwise[f"{a}_vs_{b}"] = {
                "p": p_w,
                "rank_biserial_r": r_rb
            }

        # Store results
        stats[length] = {
            "friedman_p": p_friedman,
            "kendalls_W": W,
            "pairwise": pairwise,
            "n_neurons": n_neurons
        }

    return stats


def compute_per_neuron_stats(neuron_means):
    stats = {}

    # Assume same neuron count across conditions
    n_neurons = neuron_means['prev'].shape[1]

    for neuron in range(n_neurons):

        prev_vals = neuron_means['prev'][:, neuron]
        last_vals = neuron_means['last'][:, neuron]
        next_vals = neuron_means['next'][:, neuron]

        # Remove NaNs
        prev_vals = prev_vals[~np.isnan(prev_vals)]
        last_vals = last_vals[~np.isnan(last_vals)]
        next_vals = next_vals[~np.isnan(next_vals)]

        # Kruskal omnibus test
        kw_stat, kw_p = kruskal(prev_vals, last_vals, next_vals)

        # Pairwise U tests
        pw = {}

        for a, b in [('prev','last'), ('last','next'), ('prev','next')]:

            a_vals = neuron_means[a][:, neuron]
            b_vals = neuron_means[b][:, neuron]

            a_vals = a_vals[~np.isnan(a_vals)]
            b_vals = b_vals[~np.isnan(b_vals)]

            u_stat, p_val = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
            pw[f"{a}_vs_{b}"] = p_val

        stats[neuron] = {
            'kruskal_p': kw_p,
            'pairwise': pw
        }

    return stats