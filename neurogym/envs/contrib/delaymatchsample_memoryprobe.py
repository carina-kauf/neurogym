#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CHECK https://colab.research.google.com/drive/1uryUAkHB__DUxWiR0C8uuJ3p13WeGSK2#scrollTo=fswAnFdOpbzs
# TUTORIAL https://colab.research.google.com/drive/1R2uYpcGAaC9UQuVGvrQodxM7vIzl6F9v#scrollTo=sapphire-alabama
import numpy as np
import torch

import neurogym as ngym
from neurogym import spaces

from src.data import get_splits
from torchtext.datasets import WikiText2


class DelayMatchSampleMemProbe(ngym.TrialEnv):
    r"""Delayed match-to-sample task.

    A sample stimulus is shown during the sample period. The stimulus is
    a word. After a delay period, a test stimulus is
    shown. The agent needs to determine whether the sample and the test
    stimuli are equal, and report that decision during the decision period.
    """
    metadata = {}

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)
        self.choices = [1, 2]
        self.stim_length = 5 #From how many words to choose probe

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.abort = False

        self.timing = { #this determines sequence length! (i.e., length of Trial action shape
            'fixation': 300,
            'sample1': 100,
            'sample2': 100,
            'sample3': 100,
            'sample4': 100,
            'sample5': 100,
            'delay': 1000,
            'test': 500,
            'decision': 900}
        if timing:
            self.timing.update(timing)

        self.vocab = get_splits(WikiText2, batch_size=20, return_only_vocab=True)
        self.vocab_size = len(self.vocab)

        name = {'stimulus': range(self.vocab_size+1)}
        self.observation_space = spaces.Discrete(self.vocab_size+1, name=name) #https://grid2op.readthedocs.io/en/latest/gym.html
        #longer term this should be a dictionary, but currently not supported by neurogym right now

        # dictionary observation space (can take both integers and other inputs)
        # every time I sample from observation I get a dictionary, e.g. 'image': image, 'token': token, 'rule'
        # could be in a wrapper (there's one for Box)
        #for key, val in ob.items:
            #self.encoders[key]

        name = {'fixation': 0, 'match': 1, 'non-match': 2}
        self.action_space = spaces.Discrete(3, name=name) #can take on of three actions

    def _new_trial(self, **kwargs):
        """
        _new_trial() is called when a trial ends to generate the next trial.
        Here you have to set:
        The trial periods: fixation, stimulus...
        Optionally, you can set:
        The ground truth: the correct answer for the created trial.
        """
        # Trial
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'sample_words': [self.rng.choice(range(3,self.vocab_size)) for i in range(self.stim_length)]#0 is unk, 1 and 2 are <trial_start> and <decision> and are used as stimulus start/decision period markers
        }
        #print(trial)
        trial.update(kwargs)

        ground_truth = trial['ground_truth']
        sample_words = trial['sample_words']

        if ground_truth == 1:
            memory_probe = self.rng.choice(sample_words)
        else:
            same_word_picked = True
            while same_word_picked:
                memory_probe = self.rng.choice(self.vocab_size)
                if memory_probe not in sample_words:
                    same_word_picked = False
        trial['test_word'] = memory_probe

        stim_samples = sample_words
        stim_test = memory_probe

        # Periods
        self.add_period(['fixation', 'sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'delay', 'test', 'decision'])

        stim_start, stim_end = self.vocab["<trial_start>"], self.vocab["<decision>"] #TODO ADD EOS token here instead
        self.add_ob(stim_start, 'fixation')
        self.add_ob(stim_end, 'decision')
        self.add_ob(stim_samples[0], 'sample1')
        self.add_ob(stim_samples[1], 'sample2')
        self.add_ob(stim_samples[2], 'sample3')
        self.add_ob(stim_samples[3], 'sample4')
        self.add_ob(stim_samples[4], 'sample5')
        self.add_ob(stim_test, 'test')

        self.set_groundtruth(ground_truth, 'decision')

        return trial

    def _step(self, action): #apply action to environment ##we don't need this because we're not doing RL. We do supervised learning instead.
        """
        _step receives an action and returns:
            a new observation, ob
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        reward = 0

        ob = self.ob_now
        gt = self.gt_now

        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt} #new_state, reward, end_trial, info

#
if __name__ == '__main__':
    # Instantiate the task
    env = DelayMatchSampleMemProbe()
    trial = env.new_trial()
    print('Trial info', trial)
    print('Trial observation shape', env.ob.shape)
    print('Trial action shape', env.gt.shape)
    env.reset()
    print(env.action_space)
    print("HERE", env.action_space.shape)
    print(env.gt)
    ob, reward, done, info = env.step(env.action_space.sample())
    print(type(ob))
    #ob = np.array([ob]) #Q: How can I get this shape??
    print(ob)
    print(np.shape(ob))
    print('Single time step observation shape', ob.shape)

    ob, gt = env.ob, env.gt
    ob = ob[:, np.newaxis, np.newaxis]  # Add batch axis
    inputs = torch.from_numpy(ob).type(torch.float)

    #(x,) represents the shape of your observation space. The output is ('row','column').
    # so () is integer

    dataset = ngym.Dataset(env, batch_size=20, seq_len=100)