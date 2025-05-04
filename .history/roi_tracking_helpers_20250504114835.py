from pathlib import Path
import re, os, sys
import numpy as np
import matplotlib.pyplot as plt
import roicat
import scipy.sparse
from typing import Union, Optional
import collections
import json
from roicat import helpers
# import richfile as rf
import torch 
import pandas as pd 
import pickle


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
    # ROICaT_results = load_sparse_array(path=paths_save['results_all'])
    # ROICaT_results = load_json_dict(path=paths_save['results_all'])
    # ROICaT_results = roicat.util.RichFile_ROICaT(path=paths_save['results_all'])
    ROICaT_results = RichFile_ROICaT(path=paths_save['results_all'])
    print(ROICaT_results)
    # labels_bySession = ROICaT_results['clusters']['labels_bySession'].load()
    # roi_labels = [labels_bySession[i] for i in protocol_nums]

    # ROIs = ROICaT_results['ROIs'].load()
    # rois = [ROIs['ROIs_aligned'][i] for i in protocol_nums]

    # # Only visualize the tracked neurons 
    # if tracked_neuron_ids is not None:
    #     roi_labels = [
    #         np.array(session_labels)[valid_ids]
    #         for session_labels, valid_ids in zip(roi_labels, tracked_neuron_ids)]

    #     rois = [session_labels[valid_ids] for session_labels, valid_ids in zip(rois, tracked_neuron_ids)]

    # # Find clusters
    # FOV_clusters = roicat.visualization.compute_colored_FOV(
    #     spatialFootprints=[r.power(1.0) for r in rois],  ## Spatial footprint sparse arrays
    #     FOV_height=ROIs['frame_height'],
    #     FOV_width=ROIs['frame_width'],
    #     labels=roi_labels
    # )

    # # Visualize ROIs
    # roicat.visualization.display_toggle_image_stack(
    #     FOV_clusters, 
    #     image_size=1.5,
    # )


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
class JSON_List(list):
    def __init__(self, *args, **kwargs):
        super(JSON_List, self).__init__(*args, **kwargs)

## Wrapper for SWT
class Model_SWT(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(Model_SWT, self).__init__()
        self.add_module('model', model)
    def forward(self, x):
        return self.model(x)

class RichFile_ROICaT:
    def __init__(self, path: Union[str, Path], check: bool = True):
        self.path = Path(path)
        if check and not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

    def load_all(self):
        data = {}
        for file in self.path.glob("*"):
            loader = self._get_loader(file)
            if loader:
                data[file.stem] = loader(file)
        return data

    def _get_loader(self, file: Path):
        suffix = file.suffix.lower()
        if suffix == ".npy":
            return lambda f: np.load(f, allow_pickle=True)
        elif suffix == ".npz":
            return scipy.sparse.load_npz
        elif suffix == ".json":
            return lambda f: JSON_Dict(json.load(open(f, 'r')))
        elif suffix == ".pkl":
            return lambda f: pickle.load(open(f, 'rb'))
        elif suffix == ".optuna":
            return lambda f: pickle.load(open(f, 'rb'))  # assumes saved w/ pickle
        elif suffix == ".csv":
            return pd.read_csv
        else:
            print(f"Skipping unrecognized file type: {file.name}")
            return None
        
# class RichFile_ROICaT(rf.RichFile):
    # def __init__(
    #     self,
    #     path: Optional[Union[str, Path]] = None,
    #     check: Optional[bool] = True,
    #     safe_save: Optional[bool] = True,
    # ):
    #     super().__init__(path=path, check=check, safe_save=safe_save)


    #     ## NUMPY ARRAY
    #     import numpy as np

    #     def save_npy_array(
    #         obj: np.ndarray,
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> None:
    #         """
    #         Saves a NumPy array to the given path.
    #         """
    #         np.save(path, obj, **kwargs)

    #     def load_npy_array(
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> np.ndarray:
    #         """
    #         Loads an array from the given path.
    #         """    
    #         return np.load(path, **kwargs)
        

    #     ## SCIPY SPARSE MATRIX
    #     import scipy.sparse

    #     def save_sparse_array(
    #         obj: scipy.sparse.spmatrix,
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> None:
    #         """
    #         Saves a SciPy sparse matrix to the given path.
    #         """
    #         scipy.sparse.save_npz(path, obj, **kwargs)

    #     def load_sparse_array(
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> scipy.sparse.csr_matrix:
    #         """
    #         Loads a sparse array from the given path.
    #         """        
    #         return scipy.sparse.load_npz(path, **kwargs)
        

    #     ## JSON DICT
    #     import collections
    #     import json

    #     def save_json_dict(
    #         obj: collections.UserDict,
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> None:
    #         """
    #         Saves a dictionary to the given path.
    #         """
    #         with open(path, 'w') as f:
    #             json.dump(dict(obj), f, **kwargs)

    #     def load_json_dict(
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> collections.UserDict:
    #         """
    #         Loads a dictionary from the given path.
    #         """
    #         with open(path, 'r') as f:
    #             return JSON_Dict(json.load(f, **kwargs))


    #     ## JSON LIST   
    #     def save_json_list(
    #         obj: collections.UserList,
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> None:
    #         """
    #         Saves a list to the given path.
    #         """
    #         with open(path, 'w') as f:
    #             json.dump(list(obj), f, **kwargs)

    #     def load_json_list(
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> collections.UserList:
    #         """
    #         Loads a list from the given path.
    #         """
    #         with open(path, 'r') as f:
    #             return JSON_List(json.load(f, **kwargs))
            

    #     ## OPTUNA STUDY
    #     import optuna
    #     import pickle

    #     ## load and save functions for optuna study
    #     def save_optuna_study(
    #         obj: optuna.study.Study,
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> None:
    #         """
    #         Saves an Optuna study to the given path.
    #         """
    #         with open(path, 'wb') as f:
    #             pickle.dump(obj, f, **kwargs)

    #     def load_optuna_study(
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> optuna.study.Study:
    #         """
    #         Loads an Optuna study from the given path.
    #         """
    #         with open(path, 'rb') as f:
    #             return pickle.load(f, **kwargs)
            
        
    #     ## TORCH TENSOR
    #     import torch

    #     def save_torch_tensor(
    #         obj: torch.Tensor,
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> None:
    #         """
    #         Saves a PyTorch tensor to the given path as a NumPy array.
    #         """
    #         np.save(path, obj.detach().cpu().numpy(), **kwargs)

    #     def load_torch_tensor(
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> torch.Tensor:
    #         """
    #         Loads a PyTorch tensor from the given path.
    #         """
    #         return torch.from_numpy(np.load(path, **kwargs))


    #     ## REPR
    #     def save_repr(
    #         obj: object,
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> None:
    #         """
    #         Saves the repr of an object to the given path.
    #         """
    #         with open(path, 'w') as f:
    #             f.write(repr(obj))

    #     def load_repr(
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> object:
    #         """
    #         Loads the repr of an object from the given path.
    #         """
    #         with open(path, 'r') as f:
    #             return f.read()

    #     import hdbscan

        
    #     ## PANDAS DATAFRAME
    #     import pandas as pd
        
    #     def save_pandas_dataframe(
    #         obj: pd.DataFrame,
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> None:
    #         """
    #         Saves a Pandas DataFrame to the given path.
    #         """
    #         ## Save as a CSV file
    #         obj.to_csv(path, index=True, **kwargs)

    #     def load_pandas_dataframe(
    #         path: Union[str, Path],
    #         **kwargs,
    #     ) -> pd.DataFrame:
    #         """
    #         Loads a Pandas DataFrame from the given path.
    #         """
    #         ## Load as a CSV file
    #         return pd.read_csv(path, index_col=0, **kwargs)

    #     roicat_module_tds = [rf.functions.Container(
    #         type_name=type_name,
    #         object_class=object_class,
    #         suffix="roicat",
    #         library="roicat",
    #         versions_supported=[">=1.1", "<2"],
    #     ) for type_name, object_class in [
    #         # ("data_suite2p", data_importing.Data_suite2p),
    #         # ("data_caiman", data_importing.Data_caiman),
    #         # ("data_roiextractors", data_importing.Data_roiextractors),
    #         # ("data_roicat", data_importing.Data_roicat),
    #         # ("aligner", alignment.Aligner),
    #         # ("blurrer", blurring.ROI_Blurrer),
    #         # ("roinet", ROInet.ROInet_embedder),
    #         # ("swt", scatteringWaveletTransformer.SWT),
    #         # ("similarity_graph", similarity_graph.ROI_graph),
    #         # ("clusterer", clustering.Clusterer),

    #         ("toeplitz_conv", helpers.Toeplitz_convolution2d),
    #         ("convergence_checker_optuna", helpers.Convergence_checker_optuna),
    #         ("image_alignment_checker", helpers.ImageAlignmentChecker),
    #     ]]
    #     # roicat_module_tds = []
        

    #     type_dicts = [
    #         {
    #             "type_name":          "numpy_array",
    #             "function_load":      load_npy_array,
    #             "function_save":      save_npy_array,
    #             "object_class":       np.ndarray,
    #             "suffix":             "npy",
    #             "library":            "numpy",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "numpy_scalar",
    #             "function_load":      load_npy_array,
    #             "function_save":      save_npy_array,
    #             "object_class":       np.number,
    #             "suffix":             "npy",
    #             "library":            "numpy",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "scipy_sparse_array",
    #             "function_load":      load_sparse_array,
    #             "function_save":      save_sparse_array,
    #             "object_class":       scipy.sparse.spmatrix,
    #             "suffix":             "npz",
    #             "library":            "scipy",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "json_dict",
    #             "function_load":      load_json_dict,
    #             "function_save":      save_json_dict,
    #             "object_class":       JSON_Dict,
    #             "suffix":             "json",
    #             "library":            "python",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "json_list",
    #             "function_load":      load_json_list,
    #             "function_save":      save_json_list,
    #             "object_class":       JSON_List,
    #             "suffix":             "json",
    #             "library":            "python",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "optuna_study",
    #             "function_load":      load_optuna_study,
    #             "function_save":      save_optuna_study,
    #             "object_class":       optuna.study.Study,
    #             "suffix":             "optuna",
    #             "library":            "optuna",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "torch_tensor",
    #             "function_load":      load_torch_tensor,
    #             "function_save":      save_torch_tensor,
    #             "object_class":       torch.Tensor,
    #             "suffix":             "npy",
    #             "library":            "torch",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "model_swt",
    #             "function_load":      load_repr,
    #             "function_save":      save_repr,
    #             "object_class":       Model_SWT,
    #             "suffix":             "swt",
    #             "library":            "onnx2torch",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "torch_module",
    #             "function_load":      load_repr,
    #             "function_save":      save_repr,
    #             "object_class":       torch.nn.Module,
    #             "suffix":             "torch_module",
    #             "library":            "torch",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "torch_sequence",
    #             "function_load":      load_repr,
    #             "function_save":      save_repr,
    #             "object_class":       torch.nn.Sequential,
    #             "suffix":             "torch_sequence",
    #             "library":            "torch",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "torch_dataset",
    #             "function_load":      load_repr,
    #             "function_save":      save_repr,
    #             "object_class":       torch.utils.data.Dataset,
    #             "suffix":             "torch_dataset",
    #             "library":            "torch",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "torch_dataloader",
    #             "function_load":      load_repr,
    #             "function_save":      save_repr,
    #             "object_class":       torch.utils.data.DataLoader,
    #             "suffix":             "torch_dataloader",
    #             "library":            "torch",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "hdbscan",
    #             "function_load":      load_repr,
    #             "function_save":      save_repr,
    #             "object_class":       hdbscan.HDBSCAN,
    #             "suffix":             "hdbscan",
    #             "library":            "torch",
    #             "versions_supported": [],
    #         },
    #         {
    #             "type_name":          "pandas_dataframe",
    #             "function_load":      load_pandas_dataframe,
    #             "function_save":      save_pandas_dataframe,
    #             "object_class":       pd.DataFrame,
    #             "suffix":             "csv",
    #             "library":            "pandas",
    #             "versions_supported": [],
    #         },
    #     ] + [t.get_property_dict() for t in roicat_module_tds]

    #     [self.register_type_from_dict(d) for d in type_dicts]


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
    

