import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import math

def plot(fscore, step, arch, d, prefix, s, mode):

    fig = plt.figure()
    
    ax = fig.add_subplot(111)

    summe_color = '#24d8d2'

    #tvsum_color = '#dcf7f6'
    
    tvsum_color = '#4298f4'

    d_patch = mpatches.Patch(color = '#24d8d2', label = d)

    #summe_patch = mpatches.Patch(color = summe_color, label = 'SumMe')
    
    #tvsum_patch = mpatches.Patch(color = tvsum_color, label = 'TVSum')
    
    plt.legend(handles = [d_patch])
    
    plt.plot(step, fscore, color = '#24d8d2', zorder = 0)

    x = 0

    y = 5

    ax.annotate('{:.2f}'.format(fscore[0]), xy = (step[0] + x, fscore[0] + y), ha = "center", va = "center")

    plt.scatter(step[0], fscore[0], c = summe_color, marker = 'o', zorder = 1)

    if s != 0:
    
        max_idx = np.argmax(fscore)

        if max_idx != len(fscore) - 1:
        
            ax.annotate('{:.2f}'.format(fscore[-1]), xy = (step[-1] + x, fscore[-1] + y), ha = "center", va = "center")

            plt.scatter(step[-1], fscore[-1], c = summe_color, marker = 'o', zorder = 1)
        
        ax.annotate('{:.2f}'.format(fscore[max_idx]), xy = (step[max_idx] + x, fscore[max_idx] + y), ha = "center", va = "center")

        plt.scatter(step[max_idx], fscore[max_idx], c = tvsum_color, marker = 'o', zorder = 1)
    
    #plt.plot(fscore['step'], fscore['tvsum'], color = tvsum_color)

    plt.ylim((0, 100))

    font = {'family' : 'monospace', 'weight' : 'bold'}

    plt.ylabel(mode.title(), fontdict = font)
    
    plt.xlabel('Epoch', fontdict = font)

    directory = os.path.join('figure', arch)

    if not os.path.isdir(directory):

        os.makedirs(directory)
    
    plt.savefig(os.path.join(directory, '{}_{}.png'.format(prefix, mode)))

def plot_table(f, p, r, arch, prefix, hmax = 10):
    
    videos = ['{}'.format(i + 1) for i in range(len(f))]

    batch = math.ceil(len(videos) / float(hmax))

    row_labels = ['Precision', 'Recall', 'F-Score']
    
    p2 = ['{:.2f}'.format(i * 100) for i in p]
    r2 = ['{:.2f}'.format(i * 100) for i in r]
    f2 = ['{:.2f}'.format(i * 100) for i in f]

    for b in range(batch):

        fig, ax = plt.subplots()

        # hide axes
        
        fig.patch.set_visible(False)
        
        ax.axis('off')
        
        ax.axis('tight')

        table_vals = [p2[b * hmax : (b+1) * hmax],
                      r2[b * hmax : (b+1) * hmax],
                      f2[b * hmax : (b+1) * hmax]]

        ax.table(cellText = table_vals,
                colWidths = [0.1] * len(videos[b * hmax : (b + 1) * hmax]),
                rowLabels = row_labels,
                colLabels = videos[b * hmax : (b+1) * hmax],
                #loc = 'best',
                bbox = (0.05, 0.0, 0.95, 0.32),
                rowLoc='center',
                cellLoc = 'center')

        fig.tight_layout()

        plt.savefig(os.path.join('figure', arch, 'table_{}.png'.format(b)))

        plt.clf()

def plot_fscore():

    pass


