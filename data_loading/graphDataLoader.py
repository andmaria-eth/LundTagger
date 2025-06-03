from utils.JetGraphProducer import JetGraphProducer
from torch_geometric.loader import DataLoader
from typing import Optional, List
from utils.LundTreeUtilities import OnTheFlyNormalizer
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import numpy as np
import os
import torch

class GraphDataLoader(DataLoader):
    """
    A custom DataLoader for loading JetGraphProducer objects. It allows for the
    use of a normalizer to be applied to the data, and for the specification of
    a window of the data to be loaded. It also allows for the loading of multiple
    datasets at once, and concatenates them into a single dataset.

    Args:
        dataset_paths (List[List]): A list of lists, where each inner list contains
            the path to the dataset and the name of the dataset to be loaded.
        normalizer (OnTheFlyNormalizer): A normalizer to be applied to the data.
        window_to_load (List[List[float]]): A list of lists, where each inner list
            contains the start and end of the window to be loaded. If None, the
            entire dataset is loaded.
        batch_size (int): The batch size to be used.
        dataset_fraction (List[float]): List of weights in order to have balanced datasets.
        shuffle (bool): Whether to shuffle the data.
        follow_batch (Optional[List[str]]): A list of strings specifying which
            attributes of the data object should be batched together. If None,
            the default behavior is used.
        exclude_keys (Optional[List[str]]): A list of strings specifying which
            attributes of the data object should not be batched together. If None,
            the default behavior is used.
        jet_idx (Optional(int)): An integer to filter for the corresponding jet (pt ordered) in every event
        **kwargs: Additional keyword arguments to be passed to the DataLoader.

    Returns:
        DataLoader: A DataLoader object for loading JetGraphProducer objects.
    """
    def __init__(
            self,
            dataset_paths: List[List],
            normalizer: OnTheFlyNormalizer=None,
            window_to_load: List[List[float]]=None,
            batch_size: int=1,
            dataset_fraction: List[float]=None,
            shuffle: bool = False,
            permutation: Optional[dict] = {},
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            device: torch.device=None,
            jet_idx: int = None,
            **kwargs,
    ):
        self.dataset_paths = dataset_paths
        self.normalizer = normalizer
        self.window_to_load = window_to_load
        self.shuffle = shuffle
        self.permutation = permutation
        if self.window_to_load is None:
            self.window_to_load = [[0., 1.] for _ in dataset_paths]
        self.device = device
        self.jet_idx = jet_idx

        # Consistency checks
        assert len(self.window_to_load) == len(self.dataset_paths), "Please provide one slicing window per dataset!"
        assert (not self.permutation) or (len(self.permutation) == len(self.dataset_paths)), \
            "Mismatch between number of datasets and permutations, please check for consistency"
        
        
        datasets = []
        weights = []
        # Build the dataset
        for i, (path, name) in enumerate(dataset_paths):
            if not os.path.exists(f"{path}/processed/{name}.pt"):
                raise FileNotFoundError(
                    f"The requested dataset {path}/{name} has not been processed, "\
                    "please do so before attempting to load it."
                        )
            data = JetGraphProducer(path, output_dataset_name=name)

            if self.device is not None:
                data.data.to(self.device)
            
            if self.shuffle:
                if name not in self.permutation.keys():
                    self.permutation[name] = torch.randperm(len(data))
                data = data[self.permutation[name]]
            
            data = data[int(self.window_to_load[i][0]*len(data)):int(self.window_to_load[i][1]*len(data))]
            weights.append(data.w)
            
            if self.jet_idx is not None:
                data = data[data.jet_idx == self.jet_idx]

            if dataset_fraction is not None:
                weights[i] = weights[i]*dataset_fraction[i]/np.sum(dataset_fraction)/weights[i].sum()
            
            if normalizer is not None:
                normalizer(data.data)

            datasets.append(data)

        data = ConcatDataset(datasets)
        weights = torch.cat(weights).to(device) if device is not None else torch.cat(weights)
        sampler = WeightedRandomSampler(weights, len(data), replacement=True)

        super(GraphDataLoader, self).__init__(data, batch_size, False, follow_batch, exclude_keys, sampler=sampler, **kwargs)

    def get_permutation(self):
        return self.permutation
