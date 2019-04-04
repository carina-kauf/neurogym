#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:20:01 2019

@author: linux
"""
# ['choice',
# 'stimulus',
# 'correct_side',
# 'obs_mat',
# 'act_mat',
# 'rew_mat',
# 'rep_prob']
import glob
import numpy as np
import os
print('searching here:' + os.getcwd())
files = glob.glob('Pass*npz')
files.sort(key=os.path.getmtime)
print('found files (sorted by date):')
print("\n".join(files))
choice_mat = []
stim_mat = []
r_prob_mat = []
side_mat = []
for ind_f in range(len(files)):
    data = np.load(files[ind_f])
    choice = data['choice']
    stim = data['stimulus']
    r_prob = data['rep_prob']
    side = data['correct_side']
    if choice.shape[0] != side[0]:
        dec_time = np.where(stim[:, 0] == 0)[0]
        dec_time_aux = np.concatenate((dec_time, np.array([dec_time[-1]+2])))
        dec_time_aux = np.diff(dec_time_aux)
        assert (dec_time_aux >= 1).all()
        dec_time = dec_time[dec_time_aux != 1]
        choice = choice[dec_time]
        stim = stim[dec_time-1, :]
    choice_mat.append(choice)
    stim_mat.append(stim)
    r_prob_mat.append(r_prob)
    side_mat.append(side)
    if ind_f == 0:
        SIZE = choice.shape
    assert (SIZE == choice.shape), str(SIZE) + ' ' + str(choice.shape)
    assert (SIZE[0] == stim.shape[0]), str(SIZE) + ' ' + str(stim.shape)
    assert (SIZE[0] == r_prob.shape[0]), str(SIZE) + ' ' + str(r_prob.shape)
    assert (SIZE == side.shape), str(SIZE) + ' ' + str(side.shape)

choice_mat = np.reshape(np.array(choice_mat), (SIZE*len(files), ))
stim_mat = np.reshape(np.array(stim_mat), (SIZE*len(files), stim.shape[1]))
side_mat = np.reshape(np.array(side_mat), (SIZE*len(files), ))
r_prob_mat = np.reshape(np.array(r_prob_mat), (SIZE*len(files),
                                               r_prob.shape[1]))

data = {'choice': choice_mat, 'stimulus': stim_mat,
        'correct_side': side_mat, 'rep_prob': r_prob_mat}
np.savez('bhvr_data_all.npz', **data)