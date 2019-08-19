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
    result_class = [
        'epoch',
        'stats_g/mean',
        'stats_g/std',
        'stats_o/mean',
        'stats_o/std',
        'test/episode',
        'test/mean_Q',
        'test/success_rate',
        'train/episode',
        'train/success_rate'
        ]

    all_exp = []
    num_exps = 0 
    for logfile in os.listdir(log_folder):
        csv = pd.read_csv(log_folder+logfile+'/progress.csv', skipinitialspace=True, usecols=result_class)
        exp = csv[result]._values
        exp = exp[0:n_epoch] # useful for align, bc some experiments didn't finish all epochs
        all_exp.append(exp)
        num_exps +=1
    print('total_experiments={}'.format(num_exps))

    value_down, value_media, value_up = np.percentile(all_exp, precentile, axis=0)

    plt.figure(figsize=(15,10))
    plt.plot(np.arange(len(value_media)), value_media, color='blue' )
    plt.fill_between(np.arange(len(value_media)),value_up,value_down,color='b',alpha = 0.2)

    plt.xlabel('n_epoch')
    plt.ylabel(result)
    plt.title(title)
    plt.legend(loc=2)
    plt.savefig(title+'.png')
    plt.show()

if __name__ == '__main__':
    plot()