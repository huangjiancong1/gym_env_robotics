import glob
import os
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Time New Roman"

@click.command()
@click.option('--result', type=str, default='test/success_rate', help='epoch,stats_g/mean,stats_g/std,stats_o/mean,stats_o/std,test/episode,test/mean_Q,test/success_rate,train/episode,train/success_rate')
@click.option('--log_folder', type=str, default = '/home/data/jim/her_goal_augmentation/Weight_0.7_0.3/', help='the log_path you use in baselines.run commamd')
@click.option('--precentile', type=list, default=[25,50,75], help='the precent you want to use to compare')
@click.option('--n_epoch', type=int, default=50, help='how many epoch you want to see from 0~n_epoch')
@click.option('--title', type=str, default='FetchExperiments', help='tile of the table you plotting')

def plot(result, log_folder, precentile, n_epoch, title):
    # result_class = [
    #     'epoch',
    #     'stats_g/mean',
    #     'stats_g/std',
    #     'stats_o/mean',
    #     'stats_o/std',
    #     'test/episode',
    #     'test/mean_Q',
    #     'test/success_rate',
    #     'train/episode',
    #     'train/success_rate'
    #     ]

    all_exp = []
    num_exps = 0 
    for logfile in os.listdir(log_folder):
        csv = pd.read_csv(log_folder+logfile+'/progress.csv', skipinitialspace=True)
        exp = csv[result]._values
        exp = exp[0:n_epoch] # useful for align, bc some experiments didn't finish all epochs
        all_exp.append(exp)
        num_exps +=1
    print('total_experiments={}'.format(num_exps))

    value_down, value_media, value_up = np.percentile(all_exp, precentile, axis=0)



    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, n_epoch, 20)
    minor_ticks = np.arange(0, n_epoch, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.plot(np.arange(len(value_media)), value_media, color='blue' )
    plt.fill_between(np.arange(len(value_media)),value_up,value_down,color='b',alpha = 0.2)

    plt.suptitle(title)
    ax.set_xlabel('n_epoch')
    ax.set_ylabel(result)

    plt.savefig(title+'.png')
    plt.show()

if __name__ == '__main__':
    plot()