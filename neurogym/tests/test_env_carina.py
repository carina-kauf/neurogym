import numpy as np

import gym
import neurogym as ngym
import matplotlib.pyplot as plt

def test_run(env=None, num_steps=100, verbose=False, **kwargs):
    """Test if one environment can at least be run."""
    if env is None:
        env = ngym.all_envs()[0]

    if isinstance(env, str):
        env = gym.make(env, **kwargs)
    else:
        if not isinstance(env, gym.Env):
            raise ValueError('env must be a string or a gym.Env')

    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        if done:
            env.reset()

    tags = env.metadata.get('tags', [])
    all_tags = ngym.all_tags()
    for t in tags:
        if t not in all_tags:
            print('Warning: env has tag {:s} not in all_tags'.format(t))

    if verbose:
        print(env)

    return env

def test_trialenv(env=None, **kwargs):
    """Test if a TrialEnv is behaving correctly."""
    if env is None:
        env = ngym.all_envs()[0]

    if isinstance(env, str):
        env = gym.make(env, **kwargs)
    else:
        if not isinstance(env, gym.Env):
            raise ValueError('env must be a string or a gym.Env')

    trial = env.new_trial()
    assert trial is not None, 'TrialEnv should return trial info dict ' + str(env)

def test_print(env_name):
    """Test printing of all experiments."""
    success_count = 0
    total_count = 0
    total_count += 1
    print('')
    print('Test printing env: {:s}'.format(env_name))
    env = gym.make(env_name)
    print(env)

def test_full():
    # Instantiate the task
    print("OK")
    env = gym.make("contrib.DelayMatchSampleWord-v0", **kwargs)
    trial = env.new_trial()
    print('Trial info', trial)
    print('Trial observation shape', env.ob.shape)
    print('Trial action shape', env.gt.shape)
    env.reset()
    ob, reward, done, info = env.step(env.action_space.sample())
    print('Single time step observation shape', ob.shape)

if __name__=="__main__":
    kwargs = {'dt': 20, 'timing': {'stimulus': 1000}}
    env = gym.make("contrib.DelayMatchSampleWord-v0", **kwargs)
    env.render(mode="human")
    test_run(env="contrib.DelayMatchSampleWord-v0")
    test_trialenv(env="contrib.DelayMatchSampleWord-v0")
    test_print(env_name="contrib.DelayMatchSampleWord-v0")
    test_full()
    # for i in range(5):
    #     _ = ngym.utils.plot_env(env, num_trials=2) #ValueError: ob shape (231,) not supported > cannot take integers
    #     plt.show()

    env = gym.make("DelayMatchSample-v0", **kwargs)
    test_run(env="DelayMatchSample-v0")
    test_trialenv(env="DelayMatchSample-v0")
    for i in range(5):
        _ = ngym.utils.plot_env(env, num_trials=2)
        plt.show()

    # This is a simple task, the input and output are low-dimensional
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print('Input size', input_size)
    print('Output size', output_size)

    print("Done")