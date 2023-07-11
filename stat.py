import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np

sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes

step = 5

episodes = range(100, 20001, step*100)

def plot_rewards(file):
    with open('./results/learning_curves/' + file + '.pkl', 'rb') as f:
        experiment = pickle.load(f)
    plt.plot(episodes, experiment["rewards"][::step], linewidth=2, label=file)

def plot_lr(file):
    with open('./results/learning_curves/' + file + '.pkl', 'rb') as f:
        experiment = pickle.load(f)
    plt.plot(episodes, experiment["rewards"][::step], linewidth=2, label=file)

def total_time(file):
    with open('./results/learning_curves/' + file + '.pkl', 'rb') as f:
        experiment = pickle.load(f)
    
    print(sum(experiment["times"])/60)

plt.figure()
plot_rewards('num_units_10')
plot_rewards('num_units_32')
plot_rewards('baseline2')
plot_rewards('num_units_128')
plot_rewards('3_hidden_64')

plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Learning curve: baseline')
plt.legend()


plt.figure()
plot_lr('lr_0.0001')
plot_lr('lr_0.001')
plot_lr('baseline')
plot_lr('lr_0.1')

plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Sensitivity analysis: Learning rate')
plt.legend()

plt.show()
# total_time('baseline2')
