# Provided under BSD license
# Copyright (c) 2017, Jeremy Wohlwend
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of SimExm nor the names of its contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL JEREMY WOHLWEND BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
labeling.py

Set of methods to handle the labeling of ground truth data.
Implements the brainbow method.

Ported to Python 3. Only change from the original: the print
statement in label() is now a function call. No logic, no
numerical behavior, and no library dependencies changed in this
file -- it had no dead-API imports to begin with.
"""
import numpy as np
from numpy.random import random_sample

def label(gt_dataset, volume_dim, voxel_dim, labeling_params):
    """
    Labeles the given ground truth dataset according to the config parameters.
    Creates a 3d volume for each fluorophore.

    Args:
        gt_dataset: dict cell_id -> region -> voxels
            the dictionary containing the cell data, splitted by cell_ids and cell regions,
            see load.py for more information.
        volume_dim: (z, x, y) tuplpe
            the dimensions of the ground truth dataset
        voxel_dim: (z, x, y) tuple
            the dimensions of a single voxel in nanometers
        labeling_params: dict
            dictionary containing the labeling parameters for each fluorophore
    Returns:
        labeled_volumes: dict fluorophore->3d array
            a dict from fluorophore to corresponfing 3d volume
        labeled_cells: dict fluorophore -> list of cell_ids
            list of cells labeled for each fluorophore
    """
    labeled_volumes = dict()
    labeled_cells = dict()
    #Use global density and reduce the size of gt_dataset here
    global_density = labeling_params["global_density"]
    gt_dataset = {k: v for k,v in gt_dataset.items() if random_sample() < global_density}
    #Label in the order specified in the configuration
    layers = sorted(labeling_params.keys())
    #Remove global_density
    layers.remove("global_density")
    for layer in layers:
        print("Labeling {}".format(layer))
        fluorophore = labeling_params[layer]['fluorophore']
        volume, cells  = brainbow(gt_dataset, volume_dim, voxel_dim, **labeling_params[layer])
        if fluorophore in labeled_volumes:
            labeled_volumes[fluorophore] += volume
            labeled_cells[fluorophore] |= cells
        else:
            labeled_volumes[fluorophore] = volume
            labeled_cells[fluorophore] = cells
    return labeled_volumes, labeled_cells


def brainbow(gt_dataset, volume_dim, voxel_dim, region, labeling_density,\
             protein_density, protein_noise, antibody_amp, single_neuron, **kwargs):
    """
    Distributes fluorophores and the corresponding antibodies accross the given cells.
    Follows the brainbow labeling strategy: proteins are distributed
    on the given cell region using a multinomial distribution.
    Then, fluorophore locations are computed by dstirbuting antibodies
    around the protein locations.

    Args:
        gt_dataset: dict cell_id -> region -> voxels
            ground truth dataset in dict format
        volume_dim: (z, x, y) tuple
            dimensions of the ground truth volume
        voxel_dim: (z, x, y) tuple
            dimensions of a ground truth voxel
        region: list of (z, x, y) tuples
            list of voxels
        labeling_density: float64
            the proportion of cells to label
        protein_density: float
            the density of protein labeling, in protein per nm^3
        protein_noise: float
            the amount of protein noise to include.
            Determines what proportion of proteins flies away
            from the labeled region
        antibody_amp: float
            the factor by which to amplify the protein density
        single_neuron: boolean
            if True, only a single cell is labeled

    Returns:
        labeled_volumes: numpy uint32 3D array
            the labeled volume
        labeled_cells: set
            the set of cell_ids, indicating which cells were labeled
    """
    to_label = {cell_id for cell_id in gt_dataset if random_sample() < labeling_density}
    labeled_cells = set()
    if single_neuron:
        # Get largest in the volume
        to_label = {max(to_label, key=lambda x: len(gt_dataset[x][region]))}
    #Create empty volume
    volume = np.zeros(volume_dim, np.uint32)
    for cell_id in to_label:
        #Get cell data
        reg = gt_dataset[cell_id][region]
        if len(reg) == 0: continue
        labeled_cells.add(cell_id)
        voxels = noise(reg, volume_dim, voxel_dim, protein_noise)
        #Compute number of proteins to distribute
        prob = np.ones(voxels.shape[0], np.float64) * 1.0 / voxels.shape[0]
        mean_proteins = int(protein_density * voxels.shape[0] * np.prod(voxel_dim))
        num_proteins = np.random.poisson(mean_proteins)
        distribution = np.random.multinomial(num_proteins, prob, size=1).squeeze()
        distribution = np.round(distribution * antibody_amp).astype(np.uint32)
        #add proteins to volume
        np.add.at(volume, tuple(np.transpose(voxels)), distribution)
    return volume, labeled_cells

def noise(voxels, volume_dim, voxel_dim, protein_noise):
    """
    Adds gaussian noise to a random subset of the given voxels.
    Clips values outside of the volume dimensions.

    Args:
        voxels: numpy 2D array (n x 3)
            list of (z, x, y) tuple representing a voxel
        volume_dim: (z, x, y) tuple
            dimensions of the ground truth volume
        voxel_dim: (z, x, y) tuple
            dimensions of a ground truth voxel
        protein_noise: float
            the amount of protein noise to include.
            Determines what proportion of proteins flies away
            from the labeled region
    Returns:
        voxels: numpy 2D array (n x 3)
            list of voxels including gaussian noise
    """
    ab_std = 100.0 / np.array(voxel_dim)#200 nm seems to work well as std
    gaussian = np.stack([np.random.normal(0, std, len(voxels)) for std in ab_std], axis=-1)
    #Get random subset
    non_noisy = int((1 - protein_noise) * len(voxels))
    indices = np.random.choice(len(voxels), size=non_noisy)
    #Set these to 0
    gaussian[indices, :] = [0, 0, 0]
    #Add noise
    voxels = np.round(voxels + gaussian).astype(np.int32)
    #Remove out of bounds voxels
    low, high = np.array([0, 0, 0]), np.array(volume_dim)
    inidx = np.all(np.logical_and(low <= voxels, voxels < high), axis=1)
    return voxels[inidx]
