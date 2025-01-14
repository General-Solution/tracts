#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-type conditioned to pedigree
"""

from tracts.phase_type_distribution import PhaseTypeDistribution, PhTDioecious

import numpy as np
import itertools
from joblib import Parallel, delayed
from functools import partial
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def get_pedigree(T):
    n_ind = 2 ** T - 1
    n_anc = 2 ** (T - 1)
    the_pedigree = np.zeros([n_ind, n_anc])
    the_pedigree[0, :] = 1
    k = 1
    for i in range(2, T + 1):
        len_i = 2 ** (T - i)
        n_groups_i = int(n_anc / len_i)
        for j in range(n_groups_i):
            vec_ij = np.zeros(n_anc)
            vec_ij[(len_i * j):(len_i * j + len_i)] = 1
            the_pedigree[k, :] = vec_ij
            k += 1
    return the_pedigree


def generate_trees(current_gen, max_gen, N, migrants_at_last_gen=True):
    if current_gen > max_gen:
        return [[]]

    trees = []

    if current_gen == max_gen:
        # In the last generation, nodes must have values in [1, ..., N]
        start = 1 if migrants_at_last_gen else 0
        for value in range(start, N + 1):
            trees.append([value, None, None])

    else:
        # Node can take value 0 (expand further)
        for left_subtree in generate_trees(current_gen + 1, max_gen, N, migrants_at_last_gen):
            for right_subtree in generate_trees(current_gen + 1, max_gen, N, migrants_at_last_gen):
                trees.append([0, left_subtree, right_subtree])

        # Node can take values 1, ..., N (branch stops)
        for value in range(1, N + 1):
            trees.append([value, None, None])

    return trees


def tree_to_array(tree, max_nodes):
    """Convert a tree to its array representation."""
    array = [np.nan] * max_nodes  # Start with all NaN
    queue = [(tree, 0)]  # (node, index)

    while queue:
        node, index = queue.pop(0)
        if index < max_nodes:
            if isinstance(node, list):
                array[index] = node[0]
                if node[0] == 0:  # Only expand if the node is 0
                    queue.append((node[1], 2 * index + 1))  # Left child
                    queue.append((node[2], 2 * index + 2))  # Right child
            else:
                array[index] = np.nan  # Node doesn't exist

    return array


def all_possible_trees_as_arrays(T, N, mig_at_last=True):
    max_nodes = 2 ** T - 1  # Maximum number of nodes in a complete binary tree of T generations
    trees = generate_trees(1, T, N, migrants_at_last_gen=mig_at_last)
    arrays = [tree_to_array(tree, max_nodes) for tree in trees]
    return arrays


def prob_of_pop_setting(ms, all_possible_migrations_list, migrations_at_T_list, f_migrations, m_migrations, parent_sex,
                        number_ind, number_anc, Number_pops, pedigree):
    T = int(np.log(number_ind + 1) / np.log(2))
    sex_and_gen = np.zeros([number_ind, 2])
    sex_and_gen[0, :] = [0, 1]
    k = 1

    for i in range(2, T + 1):
        len_i = 2 ** (T - i)
        n_groups_i = int(number_anc / len_i)
        for j in range(n_groups_i):
            vec_ij = np.zeros(number_anc)
            vec_ij[(len_i * j):(len_i * j + len_i)] = 1
            sex_and_gen[k, 0] = int(1 - sex_and_gen[k - 1, 0])
            sex_and_gen[k, 1] = int(i)
            k += 1
    sex_and_gen[0, :] = [parent_sex, 1]

    migrant_individuals = np.asarray(all_possible_migrations_list[ms])
    mig_setting = np.concatenate([sex_and_gen[~np.isnan(migrant_individuals), :],
                                  migrant_individuals[~np.isnan(migrant_individuals), np.newaxis],
                                  np.zeros([np.sum(~np.isnan(migrant_individuals)), 1])], axis=1)

    pfmig = mig_setting[:, 0][:, np.newaxis] * f_migrations[mig_setting[:, 1].astype(int), :]
    pmmig = (1 - mig_setting[:, 0][:, np.newaxis]) * m_migrations[mig_setting[:, 1].astype(int), :]
    pfad = mig_setting[:, 0] * (1 - np.sum(f_migrations[mig_setting[:, 1].astype(int), :], axis=1))
    pmad = (1 - mig_setting[:, 0]) * (1 - np.sum(m_migrations[mig_setting[:, 1].astype(int), :], axis=1))

    pad = (pfad + pmad) * (mig_setting[:, 2] == 0)
    pmig = np.zeros([np.sum(~np.isnan(migrant_individuals)), Number_pops])

    for j in range(Number_pops):
        pmig[:, j] = ((pfmig + pmmig) * (mig_setting[:, 2] == j + 1)[:, np.newaxis])[:, j]

    p_mig_setting = np.prod(np.sum(pmig, axis=1) + pad)

    ancestral_states = np.asarray(np.nansum(pedigree * migrant_individuals[:, np.newaxis], axis=0))
    which_ancestral = np.where(np.sum(np.abs(migrations_at_T_list - ancestral_states), axis=1) == 0)[0][0]
    return int(which_ancestral), p_mig_setting


def density_hybrid_pedigree(which_migration, migration_list, T_PED, which_pop, D_model, which_L, bins, mmat_f, mmat_m,
                            rrr_f, rrr_m, is_X_chr, is_X_chr_male):
    bins = np.asarray(bins)
    if ~np.any(np.isin(bins, which_L)):  # Add L to the bins vector
        bins = np.append(bins, which_L)
    bins = bins[bins <= which_L]  # Truncate to [0, L], where the distribution is supported
    bins = np.unique(np.sort(bins))

    ancestral_setting = np.asarray(migration_list[which_migration])
    ETL_m = None
    if np.all(ancestral_setting == which_pop + 1):
        newbins = bins
        counts_f = np.zeros(len(bins))
        counts_f[-1] = 1
        ETL_f = which_L
        counts_m = np.zeros(len(bins))
        counts_m[-1] = 1
        ETL_m = which_L
    else:
        PhT_ped = PhTDioecious(migration_matrix_f=mmat_f, migration_matrix_m=mmat_m, rho_f=rrr_f, rho_m=rrr_m,
                               sex_model=D_model, X_chromosome=is_X_chr, X_chromosome_male=is_X_chr_male,
                               TPED=T_PED, setting_TP=ancestral_setting)
        if np.all(PhT_ped.source_populations_f == which_pop):
            counts_f = np.zeros(len(bins))
            counts_f[-1] = 1
            ETL_f = which_L
        elif np.all(PhT_ped.source_populations_f != which_pop) or np.all(np.isnan(PhT_ped.source_populations_f)):
            counts_f = np.nan * bins
            ETL_f = np.nan
        else:
            newbins, counts_f, ETL_f = PhaseTypeDistribution.tractlength_histogram_windowed(PhT_ped, which_pop, bins,
                                                                                            L=which_L, density=True,
                                                                                            return_only=1, freq=False)
        if np.all(PhT_ped.source_populations_m == which_pop):
            counts_m = np.zeros(len(bins))
            counts_m[-1] = 1
            ETL_m = which_L
        elif np.all(PhT_ped.source_populations_m != which_pop) or np.all(np.isnan(PhT_ped.source_populations_m)):
            counts_m = np.nan * bins
            ETL_m = np.nan
        else:
            if not is_X_chr_male:
                newbins, counts_m, ETL_m = PhaseTypeDistribution.tractlength_histogram_windowed(PhT_ped, which_pop,
                                                                                                bins, L=which_L,
                                                                                                density=True,
                                                                                                return_only=0,
                                                                                                freq=False)
            else:
                counts_m = np.ones(len(bins))
                ETL_m = np.nan
    densities_per_p_m = counts_m  #*ped_probs_m[ped_probs_m[:,0] == which_migration, 1]
    densities_per_p_f = counts_f  #*ped_probs_f[ped_probs_f[:,0] == which_migration, 1]
    return densities_per_p_f, densities_per_p_m, which_migration, ETL_f, ETL_m


def Ped_DF_density(mig_matrix_f, mig_matrix_m, TP, L, bingrid, whichpop, Dioecious_model='DF',
                   all_possible_migrations_TP=None, rr_f=1, rr_m=1, X_chr=False, X_chr_male=False, N_cores=1,
                   freq=False):
    if not np.isin(Dioecious_model, ['DF', 'DC']):
        raise Exception('The Dioecious model must be one in DF, DC.')

    mean_mig_matrix = 0.5 * (mig_matrix_f + mig_matrix_m)
    survival_factors = np.ones(len(mean_mig_matrix))
    for generation_number in range(1, len(mean_mig_matrix)):
        survival_factors[generation_number] = survival_factors[generation_number - 1] * (
                1 - sum(mean_mig_matrix[generation_number - 1]))
    t0_proportions = np.sum((0.5 * (mig_matrix_f + mig_matrix_m)) * np.transpose(survival_factors)[:, np.newaxis],
                            axis=0)

    T = int(np.shape(mig_matrix_f)[0] - 1)  # Number of generations
    Npops = int(np.shape(mig_matrix_f)[1])
    nind_TP = 2 ** TP - 1
    nanc_TP = 2 ** (TP - 1)
    migrations_at_TP = None
    if TP > T:
        raise Exception('The pedigree must include up to T generations.')

    the_pedigree = get_pedigree(TP)

    if TP < T:
        if all_possible_migrations_TP is None:
            all_possible_migrations_TP = all_possible_trees_as_arrays(TP, Npops, mig_at_last=False)
        migrations_at_TP = np.array([i for i in itertools.product(np.arange(Npops + 1).tolist(), repeat=nanc_TP)])

    elif TP == T:
        if all_possible_migrations_TP is None:
            all_possible_migrations_TP = all_possible_trees_as_arrays(TP, Npops, mig_at_last=True)
        migrations_at_TP = np.array([i for i in itertools.product(np.arange(1, Npops + 1).tolist(), repeat=nanc_TP)])

    prob_of_pop_setting_iteration_0 = partial(prob_of_pop_setting,
                                              all_possible_migrations_list=all_possible_migrations_TP,
                                              migrations_at_T_list=migrations_at_TP, f_migrations=mig_matrix_f,
                                              m_migrations=mig_matrix_m, parent_sex=0, number_ind=nind_TP,
                                              number_anc=nanc_TP,
                                              Number_pops=Npops, pedigree=the_pedigree)

    prob_of_pop_setting_iteration_1 = partial(prob_of_pop_setting,
                                              all_possible_migrations_list=all_possible_migrations_TP,
                                              migrations_at_T_list=migrations_at_TP, f_migrations=mig_matrix_f,
                                              m_migrations=mig_matrix_m, parent_sex=1, number_ind=nind_TP,
                                              number_anc=nanc_TP,
                                              Number_pops=Npops, pedigree=the_pedigree)

    print('-------------------------------------------------------------------\n')
    print("".join(['Computing pedigrees probabilities...\n']))
    print('-------------------------------------------------------------------\n')

    prob_list_m_complete = Parallel(n_jobs=N_cores, verbose=10, prefer='processes')(
        delayed(prob_of_pop_setting_iteration_0)(i) for i in range(len(all_possible_migrations_TP)))
    data_prob_m = pd.DataFrame(prob_list_m_complete, columns=['which_ancestral', 'prob'])

    if not np.isclose(np.sum(data_prob_m.groupby(['which_ancestral']).sum().reset_index()['prob'].to_numpy()), 1):
        raise Exception('Pedigree probabilities do not sum up to one.')

    prob_list_f_complete = Parallel(n_jobs=N_cores, verbose=10, prefer='processes')(
        delayed(prob_of_pop_setting_iteration_1)(i) for i in range(len(all_possible_migrations_TP)))
    data_prob_f = pd.DataFrame(prob_list_f_complete, columns=['which_ancestral', 'prob'])

    if not np.isclose(np.sum(data_prob_f.groupby(['which_ancestral']).sum().reset_index()['prob'].to_numpy()), 1):
        raise Exception('Pedigree probabilities do not sum up to one.')

    print('-------------------------------------------------------------------\n')
    print("".join(['Done! Computing phase-type densities conditioned to each pedigree...\n']))
    print('-------------------------------------------------------------------\n')
    print(len(migrations_at_TP), "densities to compute using the", Dioecious_model, "model.\n")

    density_hybrid_iteration = partial(density_hybrid_pedigree, migration_list=migrations_at_TP, which_pop=whichpop,
                                       D_model=Dioecious_model, T_PED=TP, which_L=L,
                                       bins=bingrid, mmat_f=mig_matrix_f, mmat_m=mig_matrix_m, rrr_f=rr_f, rrr_m=rr_m,
                                       is_X_chr=X_chr, is_X_chr_male=X_chr_male)
    densities_list = Parallel(n_jobs=N_cores, verbose=10, prefer='processes')(
        delayed(density_hybrid_iteration)(i) for i in range(len(migrations_at_TP)))

    print('-------------------------------------------------------------------\n')
    print("".join(['Done!\n']))
    print('-------------------------------------------------------------------\n')

    bins = np.asarray(bingrid)
    if ~np.any(np.isin(bins, L)):  # Add L to the bins vector
        bins = np.append(bins, L)
    bins = bins[bins <= L]  # Truncate to [0, L], where the distribution is supported
    bins = np.unique(np.sort(bins))

    # Remove pedigrees that yield space states with only absorbing states

    mig_out_f = [den[2] for den in densities_list if np.any(np.isnan(den[0]))]
    mig_out_m = [den[2] for den in densities_list if np.any(np.isnan(den[1]))]

    # Re-weight pedigree probabilities
    data_prob_m = data_prob_m[~data_prob_m.which_ancestral.isin(mig_out_m)]
    prob_list_m = data_prob_m.groupby(['which_ancestral']).sum().reset_index().to_numpy()
    prob_list_m[:, 1] = prob_list_m[:, 1] / np.sum(prob_list_m[:, 1])

    data_prob_f = data_prob_f[~data_prob_f.which_ancestral.isin(mig_out_f)]
    prob_list_f = data_prob_f.groupby(['which_ancestral']).sum().reset_index().to_numpy()
    prob_list_f[:, 1] = prob_list_f[:, 1] / np.sum(prob_list_f[:, 1])
    if X_chr and X_chr_male:
        final_density = np.sum(np.array([l[0] * prob_list_f[prob_list_f[:, 0] == l[2], 1] for l in densities_list if
                                         np.isin(l[2], prob_list_f[:, 0])]), axis=0)
        Exp = np.sum(np.array([l[3] * prob_list_f[prob_list_f[:, 0] == l[2], 1] for l in densities_list if
                               np.isin(l[2], prob_list_f[:, 0])]), axis=0)
        scale = t0_proportions[whichpop] * L / Exp if freq else 1
    else:
        final_density = 0.5 * (np.sum(np.array(
            [l[0] * prob_list_f[prob_list_f[:, 0] == l[2], 1] for l in densities_list if
             np.isin(l[2], prob_list_f[:, 0])]), axis=0) + np.sum(np.array(
            [l[1] * prob_list_m[prob_list_m[:, 0] == l[2], 1] for l in densities_list if
             np.isin(l[2], prob_list_m[:, 0])]), axis=0))
        Exp = 0.5 * (np.sum(np.array([l[3] * prob_list_f[prob_list_f[:, 0] == l[2], 1] for l in densities_list if
                                      np.isin(l[2], prob_list_f[:, 0])]), axis=0) + np.sum(np.array(
            [l[4] * prob_list_m[prob_list_m[:, 0] == l[2], 1] for l in densities_list if
             np.isin(l[2], prob_list_m[:, 0])]), axis=0))
        scale = 2 * t0_proportions[whichpop] * L / Exp if freq else 1
    return bins, final_density * scale
