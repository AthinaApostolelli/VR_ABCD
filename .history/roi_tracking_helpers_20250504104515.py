from pathlib import Path
import re, os, sys
import numpy as np
import matplotlib.pyplot as plt
import roicat
import scipy.sparse
from typing import Union
import collections
import json

def roicat_align_rois(roicat_dir, roicat_data_name, sessions_to_align, basepath, animal, alignment_method='F', data=None, plot_alignment=False, savepath=''):
    '''Align the neural data according to the ROI they belong to, 
    For more details look at the ROICaT documentation https://roicat.readthedocs.io/en/latest/index.html.'''

    # Data paths ROICaT
    paths_save = {
        'results_clusters': str(Path(roicat_dir) / f'{roicat_data_name}.tracking.results_clusters.json'),
        'params_used':      str(Path(roicat_dir) / f'{roicat_data_name}.tracking.params_used.json'),
        'results_all':      str(Path(roicat_dir) / f'{roicat_data_name}.tracking.results_all.richfile'),
        'run_data':         str(Path(roicat_dir) / f'{roicat_data_name}.tracking.run_data.richfile'),
    }

    # Data paths funcimg 
    protocol_nums = [int(re.search(r'protocol-t(\d+)', s).group(1)) for s in sessions_to_align]
    func_img_path = 'funcimg/Session'
    suite2p_path = 'suite2p/plane0'

    # Find session index 
    ROICaT_results = roicat.util.RichFile_ROICaT(path=paths_save['results_all'])
    # print(ROICaT_results.keys())

    # sessions = ROICaT_results['input_data'].load()
    # sessions = [Path(p).parts[7] for p in sessions['paths_stat']]

    # Load neural data 
    if data == None:
        data = [[] for s in range(len(sessions_to_align))]
        for s, sess in enumerate(sessions_to_align):
            if alignment_method == 'F':
                datapath = basepath / animal / sess / func_img_path / suite2p_path / 'F.npy'
            elif alignment_method == 'DF_F0':
                datapath = basepath / animal / sess / func_img_path / suite2p_path / 'DF_F0.npy'
                if not os.path.exists(datapath):
                    raise FileNotFoundError('The DF_F0.npy file does not exist in this directory.')
            else:
                raise KeyError('This is not a valid data format for ROI alignment.')
            data[s] = np.load(datapath)

    # Load UCIDs 
    labels_bySession = ROICaT_results['clusters']['labels_bySession'].load()
    roi_labels = [rois for rois in labels_bySession[protocol_nums[0]:protocol_nums[-1]+1]]  

    # Update UCIDs with valid cells
    iscell = [[] for s in range(len(sessions_to_align))]
    for s, sess in enumerate(sessions_to_align):
        datapath = basepath / animal / sess / func_img_path / suite2p_path / 'iscell.npy'
        iscell[s] = np.load(datapath)[:,0]

    # Apply the mask to the aligned data
    labels_iscell = roicat.util.mask_UCIDs_with_iscell(
        ucids=roi_labels,
        iscell=iscell
    )
    # Squeeze the labels to remove the unassigned labels (not necessary, but reduces the number of unique labels)
    labels_iscell = roicat.util.squeeze_UCID_labels(ucids=labels_iscell, return_array=True)  ## [(n_rois,)] * n_sessions

    # Align the data with the masked labels
    data_aligned_masked, idx_original_aligned = roicat.util.match_arrays_with_ucids(
        arrays=data,  ## expects list (length n_sessions) of numpy arrays (shape (n_rois, n_timepoints))
        ucids=labels_iscell,  ## expects list (length n_sessions) of numpy arrays (shape (n_rois,))  OR   concatenated numpy array (shape (n_rois_total,))
        return_indices=True
    )

    # Visualize the alignment
    if plot_alignment:
        fig, axs = plt.subplots(2, 2, figsize=(15, 5))
        for i in range(2):
            axs[0, i].imshow(data[i], aspect="auto", cmap="rainbow", interpolation="none")
            axs[1, i].imshow(data_aligned_masked[i], aspect="auto", cmap="rainbow", interpolation="none")
            axs[0, i].set_title(f"Session {i+1}")
            axs[1, i].set_title(f"Session {i+1} (aligned)")
            axs[0, i].set_xlabel("Timepoints")
            axs[1, i].set_xlabel("Timepoints")
            axs[0, i].set_ylabel("ROIs")
            axs[1, i].set_ylabel("ROIs")
            ## Colorbar
            if i == 2 - 1:
                fig.colorbar(axs[0, i].imshow(data[i], aspect="auto", cmap="rainbow", interpolation="none"), ax=axs[0, i], label="ROI label")
                fig.colorbar(axs[1, i].imshow(data_aligned_masked[i], aspect="auto", cmap="rainbow", interpolation="none"), ax=axs[1, i], label="ROI label")
        plt.tight_layout()

    # Save indices of aligned data 
    if savepath == '':
        savepath = os.path.join(basepath, animal)
    else:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    filename = f"roicat_aligned_ROIs_{'_'.join(['t' + str(n) for n in protocol_nums])}.npy"
    np.save(os.path.join(savepath, filename), idx_original_aligned)

    return data_aligned_masked, idx_original_aligned


# def roicat_track_rois(): 
#     # TODO 

def roicat_visualize_tracked_rois(roicat_dir, roicat_data_name, sessions_to_align, tracked_neuron_ids=None):
    '''Visualize the alignment of tracked ROIs. For more details look at the ROICaT documentation https://roicat.readthedocs.io/en/latest/index.html.'''

    # Handle single or multiple sessions
    if isinstance(sessions_to_align, str):
        sessions_to_align = [sessions_to_align]

    # Now this will always work
    protocol_nums = [int(re.search(r'protocol-t(\d+)', s).group(1)) for s in sessions_to_align]

    # Data paths ROICaT
    paths_save = {
        'results_clusters': str(Path(roicat_dir) / f'{roicat_data_name}.tracking.results_clusters.json'),
        'params_used':      str(Path(roicat_dir) / f'{roicat_data_name}.tracking.params_used.json'),
        'results_all':      str(Path(roicat_dir) / f'{roicat_data_name}.tracking.results_all.richfile'),
        'run_data':         str(Path(roicat_dir) / f'{roicat_data_name}.tracking.run_data.richfile'),
    }

    # Find session index 
    ROICaT_results = load_sparse_array(path=paths_save['results_all'])
    # ROICaT_results = roicat.util.RichFile_ROICaT(path=paths_save['results_all'])

    labels_bySession = ROICaT_results['clusters']['labels_bySession'].load()
    roi_labels = [labels_bySession[i] for i in protocol_nums]

    ROIs = ROICaT_results['ROIs'].load()
    rois = [ROIs['ROIs_aligned'][i] for i in protocol_nums]

    # Only visualize the tracked neurons 
    if tracked_neuron_ids is not None:
        roi_labels = [
            np.array(session_labels)[valid_ids]
            for session_labels, valid_ids in zip(roi_labels, tracked_neuron_ids)]

        rois = [session_labels[valid_ids] for session_labels, valid_ids in zip(rois, tracked_neuron_ids)]

    # Find clusters
    FOV_clusters = roicat.visualization.compute_colored_FOV(
        spatialFootprints=[r.power(1.0) for r in rois],  ## Spatial footprint sparse arrays
        FOV_height=ROIs['frame_height'],
        FOV_width=ROIs['frame_width'],
        labels=roi_labels
    )

    # Visualize ROIs
    roicat.visualization.display_toggle_image_stack(
        FOV_clusters, 
        image_size=1.5,
    )


def load_sparse_array(
    path: Union[str, Path],
    **kwargs,
) -> scipy.sparse.csr_matrix:
    """
    Loads a sparse array from the given path.
    """        
    return scipy.sparse.load_npz(path, **kwargs)


def load_json_dict(
    path: Union[str, Path],
    **kwargs,
) -> collections.UserDict:
    """
    Loads a dictionary from the given path.
    """
    with open(path, 'r') as f:
        return JSON_Dict(json.load(f, **kwargs))
    

class JSON_Dict(dict):
    def __init__(self, *args, **kwargs):
        super(JSON_Dict, self).__init__(*args, **kwargs)
        
          
if __name__ == '__main__':

    data_aligned_masked, idx_original_aligned = roicat_align_rois(roicat_dir=r'/Users/athinaapostolelli/Documents/SWC/VR_ABCD/ROICaT',
                      roicat_data_name='TAA0000066',
                      sessions_to_align=['ses-011_date-20250315_protocol-t5', 'ses-012_date-20250318_protocol-t6'],
                      basepath=Path('/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2'),
                      animal='TAA0000066',
                      alignment_method='DF_F0',
                      data=None,
                      plot_alignment=False,
                      savepath='')
    

