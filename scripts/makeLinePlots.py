import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

# Read in data
jf_combo = pd.read_csv('../results/combo_Duration_nIter=50_rank=200_em_summary_frenchjapanese.csv')
jf_wf = pd.read_csv('../results/wf_Duration_nIter=50_rank=200_em_summary_frenchjapanese.csv')
dutchads_combo = pd.read_csv('../results/combo_Duration_nIter=50_rank=200_em_summary_dutchads.csv')
dutchads_wf = pd.read_csv('../results/wf_Duration_nIter=50_rank=200_em_summary_dutchads.csv')
dutchids_combo = pd.read_csv('../results/combo_Duration_nIter=50_rank=5_em_summary_dutchids.csv')
dutchids_wf = pd.read_csv('../results/wf_Duration_nIter=50_rank=5_em_summary_dutchids.csv')

# Update x-axis labels
jf_combo['Input Size'] = [str(in_size) + '\nc=' + str(frv) for (in_size, frv) in zip(jf_combo['InputSize'], jf_combo['FreqRankVal'])]
jf_wf['Input Size'] = [str(in_size) + '\nc=' + str(frv) for (in_size, frv) in zip(jf_wf['InputSize'], jf_wf['FreqRankVal'])]
dutchads_combo['Input Size'] = [str(in_size) + '\nc=' + str(frv) for (in_size, frv) in zip(dutchads_combo['InputSize'], dutchads_combo['FreqRankVal'])]
dutchads_wf['Input Size'] = [str(in_size) + '\nc=' + str(frv) for (in_size, frv) in zip(dutchads_wf['InputSize'], dutchads_wf['FreqRankVal'])]
dutchids_combo['Input Size'] = [str(in_size) + '\nc=' + str(frv) for (in_size, frv) in zip(dutchids_combo['InputSize'], dutchids_combo['FreqRankVal'])]
dutchids_wf['Input Size'] = [str(in_size) + '\nc=' + str(frv) for (in_size, frv) in zip(dutchids_wf['InputSize'], dutchids_wf['FreqRankVal'])]

# Set up grid for Figure 3
rows = ['Japanese vs. French', 'Dutch ADS', 'Dutch IDS']
pad = 5 # in points

# possible values of depmeasure: Maximum, Mean
# depmeasure = 'Mean'
depmeasure = 'Maximum'
fig, axes = plt.subplots(3, 2, figsize=(11.2,11), sharey='all', sharex='all')
# sns.set(font_scale=0.9)

for ax, row in zip(axes[:,0], rows):
    ax.annotate(row,xy=(0, 0.5), xytext=(-ax.yaxis.labelpad-pad,0),                    
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation = 90, fontweight = 'bold', fontsize = 14)

fig.tight_layout(pad = 3, h_pad = 0.5, w_pad = 0.5)
# fig.subplots_adjust(left=0.075, top=0.97, bottom = 0.03)

# Set axis labels
axes[0, 0].set_title("P+NS+VQ", fontweight = 'bold', fontsize = 14)
axes[0, 1].set_title("Word Frames", fontweight = 'bold', fontsize = 14)

# Make individual plots in grid
c = sns.pointplot(ax=axes[1,0], x='Input Size', y=depmeasure, hue='ContrastType', data=dutchads_combo)
c.set(xlabel='')
d = sns.pointplot(ax=axes[1,1], x='Input Size', y=depmeasure, hue='ContrastType', data=dutchads_wf)
d.set(xlabel='', ylabel = '')
e = sns.pointplot(ax=axes[2,0], x='Input Size', y=depmeasure, hue='ContrastType', data=dutchids_combo)
f = sns.pointplot(ax=axes[2,1], x='Input Size', y=depmeasure, hue='ContrastType', data=dutchids_wf)
f.set(ylabel = '')
a = sns.pointplot(ax=axes[0,0], x='Input Size', y=depmeasure, hue='ContrastType', data=jf_combo)
a.set(xlabel='')
# a.tick_params(labelbottom=True)
b = sns.pointplot(ax=axes[0,1], x='Input Size', y=depmeasure, hue='ContrastType', data=jf_wf)
b.set(xlabel='', ylabel = '')

plt.show()
# plt.savefig('../figures/lineplot_Duration_nIter=50_rank=200_' + 'measure=' + depmeasure + '.pdf', bbox_inches = "tight", pad_inches = 0.005)
