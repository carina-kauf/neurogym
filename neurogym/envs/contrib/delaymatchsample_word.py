#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

import neurogym as ngym
from neurogym import spaces

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torchtext.datasets import WikiText2

def get_vocab(dataset): #has to be here too due to circular import otherwise
    """
    Args:
        dataset: Name of torchtext dataset

    Returns:
        vocab_size: int, size of vocabulary
        train_data: Tensor, shape [full_seq_len, batch_size], i.e., [N // bsz, bsz]
        val_data: Tensor, shape [full_seq_len, eval_batch_size (here:10)], i.e., [N // 10, 10]
        test_data: Tensor, shape [full_seq_len, eval_batch_size (here:10)], i.e., [N // 10, 10]
    """

    train_iter = dataset(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    vocab_size = len(vocab)

    return vocab, vocab_size



class DelayMatchSampleWord(ngym.TrialEnv):
    r"""Delayed match-to-sample task.

    A sample stimulus is shown during the sample period. The stimulus is
    a word. After a delay period, a test stimulus is
    shown. The agent needs to determine whether the sample and the test
    stimuli are equal, and report that decision during the decision period.
    """
    metadata = {}

    def __init__(self, dt=100, rewards=None, sigma=1, timing=None):
        super().__init__(dt=dt)
        self.choices = [1, 2]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise #TODO not sure we want that

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.abort = False

        self.timing = { #this determines sequence length! (i.e., length of Trial action shape
            'fixation': 300,
            'sample': 500,
            'delay': 1000,
            'test': 500,
            'decision': 900}
        if timing:
            self.timing.update(timing)

        self.vocab, self.vocab_size = get_vocab(WikiText2)

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

        try:
            stim_start, stim_delay, stim_decide = self.vocab["fix"], self.vocab["<unk>"], self.vocab["decide"]
        except:
            raise ValueError("Required words not found in the vocabulary!")
        exclude_from_choice = [stim_start, stim_delay, stim_decide]
        word_choice_list = list(set(list(range(self.vocab_size))) - set(exclude_from_choice))
        # Trial
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'sample_word': self.rng.choice(word_choice_list) #0 is unk, 1 and 2 are <trial_start> and <decision> and are used as stimulus start/decision period markers
        }
        #print(trial)
        trial.update(kwargs)

        ground_truth = trial['ground_truth']
        sample_word_idx = trial['sample_word']


        if ground_truth == 1:
            test_word_idx = sample_word_idx
        else:
            same_word_picked = True
            while same_word_picked:
                test_word_idx = self.rng.choice(self.vocab_size)
                if test_word_idx != sample_word_idx:
                    same_word_picked = False
        trial['test_word'] = test_word_idx

        stim_sample = sample_word_idx
        stim_test = test_word_idx

        # Periods
        self.add_period(['fixation', 'sample', 'delay', 'test', 'decision'])

        self.add_ob(stim_start, 'fixation')
        self.add_ob(stim_sample, 'sample')
        self.add_ob(stim_delay, 'delay')
        self.add_ob(stim_test, 'test')
        self.add_ob(stim_decide, 'decision')

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
    env = DelayMatchSampleWord()
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