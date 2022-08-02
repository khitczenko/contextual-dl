import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

directory = '../results/'

## Function used to prepare graphs
def prepare_df_graph(filename, context):
	lang = filename.split("_")[-1].split(".csv")[0]
	if lang == 'frenchjapanese':
		langdesc = 'Japanese vs. French'
		label = 'Japanese/French\n' + context
	elif lang == 'dutchads':
		langdesc = 'Dutch (ADS)'
		label = 'Dutch(ADS)\n' + context
	elif lang == 'dutchids':
		langdesc = 'Dutch (IDS)'
		label = 'Dutch(IDS)\n' + context
	else:
		print("Unrecognized language")

	d = pd.read_csv(filename)
	d['Language'] = [langdesc]*len(d['Status'])
	d['Contexts'] = [context]*len(d['Status'])
	d[''] = [label]*len(d['Status'])
	return(d)

########## FIGURE 2 ##########
a = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_contrast_frenchjapanese.csv', 'P+NS+VQ')
a2 = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_nocontrast_frenchjapanese.csv', 'P+NS+VQ')
b = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_contrast_frenchjapanese.csv', 'Word Frames')
b2 = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_nocontrast_frenchjapanese.csv', 'Word Frames')
c = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_contrast_dutchads.csv', 'P+NS+VQ')
c2 = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_nocontrast_dutchads.csv', 'P+NS+VQ')
d = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_contrast_dutchads.csv', 'Word Frames')
d2 = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_nocontrast_dutchads.csv', 'Word Frames')
e = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=5_em_contrast_dutchids.csv', 'P+NS+VQ')
e2 = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=5_em_nocontrast_dutchids.csv', 'P+NS+VQ')
f = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=5_em_contrast_dutchids.csv', 'Word Frames')
f2 = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=5_em_nocontrast_dutchids.csv', 'Word Frames')

# Figure 2: showing all boxplots side by side
alldfs = pd.concat([a, a2, b, b2, c, c2, d, d2, e, e2, f, f2])
alldfs['Earthmover\'s Distance'] = alldfs['EM']
ax = sns.boxplot(x = '', y = 'Earthmover\'s Distance', hue = 'Status', data = alldfs)
plt.tight_layout()
plt.show()

##############################


########## FIGURE 4 ##########

# Word Frames, VQ, NS, P, P+NS+VQ

jf_ns_contrast = prepare_df_graph(directory + 'ns_Duration_nIter=1_rank=200_em_contrast_frenchjapanese.csv', 'NS')
jf_ns_nocontrast = prepare_df_graph(directory + 'ns_Duration_nIter=1_rank=200_em_nocontrast_frenchjapanese.csv', 'NS')
jf_p_contrast = prepare_df_graph(directory + 'p_Duration_nIter=1_rank=200_em_contrast_frenchjapanese.csv', 'P')
jf_p_nocontrast = prepare_df_graph(directory + 'p_Duration_nIter=1_rank=200_em_nocontrast_frenchjapanese.csv', 'P')
jf_vq_contrast = prepare_df_graph(directory + 'vq_Duration_nIter=1_rank=200_em_contrast_frenchjapanese.csv', 'VQ')
jf_vq_nocontrast = prepare_df_graph(directory + 'vq_Duration_nIter=1_rank=200_em_nocontrast_frenchjapanese.csv', 'VQ')
dads_ns_contrast = prepare_df_graph(directory + 'ns_Duration_nIter=1_rank=200_em_contrast_dutchads.csv', 'NS')
dads_ns_nocontrast = prepare_df_graph(directory + 'ns_Duration_nIter=1_rank=200_em_nocontrast_dutchads.csv', 'NS')
dads_p_contrast = prepare_df_graph(directory + 'p_Duration_nIter=1_rank=200_em_contrast_dutchads.csv', 'P')
dads_p_nocontrast = prepare_df_graph(directory + 'p_Duration_nIter=1_rank=200_em_nocontrast_dutchads.csv', 'P')
dads_vq_contrast = prepare_df_graph(directory + 'vq_Duration_nIter=1_rank=200_em_contrast_dutchads.csv', 'VQ')
dads_vq_nocontrast = prepare_df_graph(directory + 'vq_Duration_nIter=1_rank=200_em_nocontrast_dutchads.csv', 'VQ')
dids_ns_contrast = prepare_df_graph(directory + 'ns_Duration_nIter=1_rank=200_em_contrast_dutchids.csv', 'NS')
dids_ns_nocontrast = prepare_df_graph(directory + 'ns_Duration_nIter=1_rank=200_em_nocontrast_dutchids.csv', 'NS')
dids_p_contrast = prepare_df_graph(directory + 'p_Duration_nIter=1_rank=200_em_contrast_dutchids.csv', 'P')
dids_p_nocontrast = prepare_df_graph(directory + 'p_Duration_nIter=1_rank=200_em_nocontrast_dutchids.csv', 'P')
dids_vq_contrast = prepare_df_graph(directory + 'vq_Duration_nIter=1_rank=200_em_contrast_dutchids.csv', 'VQ')
dids_vq_nocontrast = prepare_df_graph(directory + 'vq_Duration_nIter=1_rank=200_em_nocontrast_dutchids.csv', 'VQ')


# FIG 4b: JUST NEIGHBORING SOUNDS
alldfs = pd.concat([jf_ns_contrast, jf_ns_nocontrast, dads_ns_contrast, dads_ns_nocontrast,	dids_ns_contrast, dids_ns_nocontrast])
alldfs['Earthmover\'s Distance'] = alldfs['EM']
ax = sns.boxplot(x = '', y = 'Earthmover\'s Distance', hue = 'Status', data = alldfs)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# FIG 4a: JUST PROSODY
alldfs = pd.concat([jf_p_contrast, jf_p_nocontrast, dads_p_contrast, dads_p_nocontrast,	dids_p_contrast, dids_p_nocontrast])
alldfs['Earthmover\'s Distance'] = alldfs['EM']
ax = sns.boxplot(x = '', y = 'Earthmover\'s Distance', hue = 'Status', data = alldfs)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# FIG 4c: JUST VOWEL QUALITY
alldfs = pd.concat([jf_vq_contrast, jf_vq_nocontrast, dads_vq_contrast, dads_vq_nocontrast,	dids_vq_contrast, dids_vq_nocontrast])
alldfs['Earthmover\'s Distance'] = alldfs['EM']
ax = sns.boxplot(x = '', y = 'Earthmover\'s Distance', hue = 'Status', data = alldfs)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

##############################

########## FIGURE 5 ##########
# Figure of combo w/ noisy consonant class

jf_noisycombonsclass_contrast = prepare_df_graph(directory + 'noisycombonsclass_Duration_nIter=1_rank=200_em_contrast_frenchjapanese.csv', 'P(noisy)+NS(class)+VQ')
jf_noisycombonsclass_nocontrast = prepare_df_graph(directory + 'noisycombonsclass_Duration_nIter=1_rank=200_em_nocontrast_frenchjapanese.csv', 'P(noisy)+NS(class)+VQ')
dads_noisycombonsclass_contrast = prepare_df_graph(directory + 'noisycombonsclass_Duration_nIter=1_rank=200_em_contrast_dutchads.csv', 'P(noisy)+NS(class)+VQ')
dads_noisycombonsclass_nocontrast = prepare_df_graph(directory + 'noisycombonsclass_Duration_nIter=1_rank=200_em_nocontrast_dutchads.csv', 'P(noisy)+NS(class)+VQ')
dids_noisycombonsclass_contrast = prepare_df_graph(directory + 'noisycombonsclass_Duration_nIter=1_rank=5_em_contrast_dutchids.csv', 'P(noisy)+NS(class)+VQ')
dids_noisycombonsclass_nocontrast = prepare_df_graph(directory + 'noisycombonsclass_Duration_nIter=1_rank=5_em_nocontrast_dutchids.csv', 'P(noisy)+NS(class)+VQ')

alldfs = pd.concat([jf_noisycombonsclass_contrast, jf_noisycombonsclass_nocontrast, dads_noisycombonsclass_contrast, dads_noisycombonsclass_nocontrast, dids_noisycombonsclass_contrast, dids_noisycombonsclass_nocontrast])
alldfs['Earthmover\'s Distance'] = alldfs['EM']
ax = sns.boxplot(x = '', y = 'Earthmover\'s Distance', hue = 'Status', data = alldfs)
plt.tight_layout()
plt.show()

##############################

########## FIGURE 6 ##########

a = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_contrast_frenchjapanese.csv', 'P+NS+VQ')
b = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_nolongvowels_frenchjapanese.csv', 'P+NS+VQ')
c = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_nocontrast_frenchjapanese.csv', 'P+NS+VQ')
d = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_contrast_frenchjapanese.csv', 'Word Frames')
e = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_nolongvowels_frenchjapanese.csv', 'Word Frames')
f = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_nocontrast_frenchjapanese.csv', 'Word Frames')
g = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_contrast_dutchads.csv', 'P+NS+VQ')
h = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_nolongvowels_dutchads.csv', 'P+NS+VQ')
j = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_em_nocontrast_dutchads.csv', 'P+NS+VQ')
k = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_contrast_dutchads.csv', 'Word Frames')
l = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_nolongvowels_dutchads.csv', 'Word Frames')
m = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_em_nocontrast_dutchads.csv', 'Word Frames')
n = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=5_em_contrast_dutchids.csv', 'P+NS+VQ')
o = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=5_em_nolongvowels_dutchids.csv', 'P+NS+VQ')
p = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=5_em_nocontrast_dutchids.csv', 'P+NS+VQ')
q = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=5_em_contrast_dutchids.csv', 'Word Frames')
r = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=5_em_nolongvowels_dutchids.csv', 'Word Frames')
s = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=5_em_nocontrast_dutchids.csv', 'Word Frames')

alldfs = pd.concat([a, c, b, d, e, f, g, h, j, k, l, m, n, o, p, q, r, s])
alldfs['Earthmover\'s Distance'] = alldfs['EM']

ax = sns.boxplot(x = '', y = 'Earthmover\'s Distance', hue = 'Status', data = alldfs)
# ax = sns.violinplot(x = '', y = 'Earthmover\'s Distance', hue = 'Status', data = alldfs)
# ax = sns.boxenplot(x = '', y = 'Earthmover\'s Distance', hue = 'Status', data = alldfs)
# handles, labels = ax.get_legend_handles_labels()
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


##############################

######### FIGURE S2 ##########
# This generates plots looking at KL divergences instead of Earthmover's Distance

jf_combokl_contrast = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_kl_contrast_frenchjapanese.csv', 'P+NS+VQ')
jf_combokl_nocontrast = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_kl_nocontrast_frenchjapanese.csv', 'P+NS+VQ')
jf_wfkl_contrast = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_kl_contrast_frenchjapanese.csv', 'Word Frames')
jf_wfkl_nocontrast = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_kl_nocontrast_frenchjapanese.csv', 'Word Frames')
dads_combokl_contrast = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_kl_contrast_dutchads.csv', 'P+NS+VQ')
dads_combokl_nocontrast = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=200_kl_nocontrast_dutchads.csv', 'P+NS+VQ')
dads_wfkl_contrast = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_kl_contrast_dutchads.csv', 'Word Frames')
dads_wfkl_nocontrast = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=200_kl_nocontrast_dutchads.csv', 'Word Frames')
dids_combokl_contrast = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=5_kl_contrast_dutchids.csv', 'P+NS+VQ')
dids_combokl_nocontrast = prepare_df_graph(directory + 'combo_Duration_nIter=1_rank=5_kl_nocontrast_dutchids.csv', 'P+NS+VQ')
dids_wfkl_contrast = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=5_kl_contrast_dutchids.csv', 'Word Frames')
dids_wfkl_nocontrast = prepare_df_graph(directory + 'wf_Duration_nIter=1_rank=5_kl_nocontrast_dutchids.csv', 'Word Frames')

alldfs = pd.concat([jf_combokl_contrast, 
	jf_combokl_nocontrast, 
	jf_wfkl_contrast, 
	jf_wfkl_nocontrast, 
	dads_combokl_contrast, 
	dads_combokl_nocontrast, 
	dads_wfkl_contrast, 
	dads_wfkl_nocontrast, 
	dids_combokl_contrast, 
	dids_combokl_nocontrast,
	dids_wfkl_contrast, 
	dids_wfkl_nocontrast])

# Comment this code out to not make this graph!
alldfs['KL Divergence'] = alldfs['EM']
# ax = sns.violinplot(x = '', y = 'KL Divergence', hue = 'Status', data = alldfs)# cut = 0) # bw = 0.5
# ax = sns.boxenplot(x = '', y = 'KL Divergence', hue = 'Status', data = alldfs)# cut = 0) # bw = 0.5
## bw = changing this parameter adjusts the extent of smoothing
## cut = changing this to 0 
ax = sns.boxplot(x = '', y = 'KL Divergence', hue = 'Status', data = alldfs)
plt.tight_layout()
# plt.legend('') # this line gets rid of the legend if desirable
plt.legend(loc='upper left') # this line moves the legend to the top left where it's out of the way
plt.show()

##############################
