import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.stats import wasserstein_distance
from collections import defaultdict
from random import randrange
import random
import numpy as np
from scipy.stats import rankdata
from collections import defaultdict
import sys
sys.path.append('supportingfiles')
# from knn_divergence import naive_estimator as kl_divergence
from divergence_measures import symmetric_kl_divergence
# from divergence_measures import js_divergence
# from divergence_measures import hellinger_distance

def analyzeEMs(ems_by_cpair, longprop_by_context, xlabel):
	em = []
	diffs = []
	for k, v in ems_by_cpair.items():
		em.append(v)
		c1long = longprop_by_context[k[0]]
		c2long = longprop_by_context[k[1]]
		diffs.append(max(c1long,c2long))

		# if abs(c1long-c2long) > 0.95:
		# 	print(k)G
		# 	print(c1long)
		# 	print(c2long)

	ax = sns.scatterplot(x=diffs, y=em)
	plt.xlabel(xlabel)
	# plt.xlabel('Difference in Long Vowel Proportion')
	plt.ylabel('Earthmover\'s Distance')
	plt.show()


# Takes in a dictionary, allcontexts, that has CONTEXTS as keys 
# and FREQUENCIES as values
def getFrequencyRank(allcontexts):
	frequency_ranks = defaultdict(int)

	contexts = []
	freqs = []

	# Loop through allcontexts dictionaries, adding context and frequency to 
	# corresponding indices of different lists
	for k, v in allcontexts.items():
		contexts.append(k)
		freqs.append(-1*v)

	# print(contexts)
	# Use scipy.stats.rankdata to get rank for each context
	ranks = rankdata(freqs, method = 'min')

	# Add these ranks to new dictionary, frequency_ranks, which has contexts as keys
	# and ranks as values
	for ii in range(0, len(contexts)):
		frequency_ranks[contexts[ii]] = ranks[ii]

	return(frequency_ranks)

def addContextualNoise(df, columns2randomize, error_rate):
	# Loop through all columns we want to randomize
	for col in columns2randomize:
		col_list = df[col].tolist()
		# Get list of all values this column has from the corpus
		all_poss_values = np.unique(col_list).tolist()
		# Loop through and change to a different value with error rate of error_rate
		for ii in range(0, len(col_list)):
			if random.random() <= error_rate:
				current = col_list[ii]
				# print(other_poss_values)
				all_poss_values.remove(col_list[ii])
				# print(other_poss_values)
				col_list[ii] = np.random.choice(all_poss_values)
				all_poss_values.append(current)
		new_col = col + "_Noisy"
		# new_col = col + "_Error=" + str(error_rate*100)
		# print(new_col)
		df[new_col] = col_list

	return(df)

def fixProsody(df):
	wi = df['WordInitial_Noisy'].tolist()
	wf = df['WordFinal_Noisy'].tolist()
	ui = df['UttInitial_Noisy'].tolist()
	uf = df['UttFinal_Noisy'].tolist()

	for ii in range(0, len(wi)):
		if ui[ii] == 1:
			wi[ii] = 1
		if uf[ii] == 1:
			wf[ii] = 1

	df['WordInitial_Noisy'] = wi
	df['WordFinal_Noisy'] = wf

	return(df)

# Makes a dictionary to help map segments to their class
def seg2class(lang):

	mapping_japanese = {}
	mapping_french = {}
	mapping_dutch = {}

	mapping_japanese['stop'] = ['b', 'by', 'd', 'dy', 'g', 'gj', 'gy', 'k', 'kj', 'kw', 'ky', 'p', 'py', 't', 'ty']
	mapping_japanese['nasal'] = ['m', 'my', 'n', 'nj', 'ny', 'N', 'VN']
	mapping_japanese['vowel'] = ['a', 'e', 'i', 'o', 'u']
	mapping_japanese['fricative'] = ['s', 'sj', 'sy', 'v', 'z', 'zj', 'zy', 'h', 'hj', 'hy', 'c', 'cj', 'cy']
	mapping_japanese['approximant'] = ['r', 'ry', 'w', 'y']
	mapping_japanese['eow'] = ['#', '?', 'F', 'Fy']

	mapping_french['stop'] = ['p', 'b', 't', 'd', 'k', 'g']
	mapping_french['nasal'] = ['m', 'n', 'N']
	mapping_french['vowel'] = ['a', 'i', 'x', 'e', 'u', 'A', 'E', 'I', 'O', 'o', 'y']
	mapping_french['fricative'] = ['S', 'Z', 'f', 'v', 's', 'z']
	mapping_french['approximant'] = ['w', 'l', 'r', 'j', 'h']
	mapping_french['eow'] = ['#']

	mapping_dutch['stop'] = ['b', 'd', 'g', 'gc', 'k', 'p', 't', 'gh']
	mapping_dutch['nasal'] = ['m', 'n', 'nc', '@']
	mapping_dutch['vowel'] = ['@', 'a', 'ac', 'au', 'e', 'ec', 'ei', 'eu', 'i', 'ic', 'o', 'oc', 'u', 'ui', 'y', 'yc']
	mapping_dutch['fricative'] = ['f', 's', 'sc', 'v', 'z', 'h', 'x']
	mapping_dutch['approximant'] = ['j', 'l', 'r', 'w', 'er']
	mapping_dutch['eow'] = ['#']

	if lang == 'JapaneseFrench':
		return(mapping_japanese, mapping_french)
	elif lang in ['DutchADS', 'DutchIDS']:
		return(mapping_dutch, mapping_dutch)
	else:
		print("Unrecognized language")
		return

def nsClass(df, mapping):
	for col in ['PrevSeg', 'FollSeg', 'ModPrevSeg', 'ModFollSeg']:
		newcol = []
		col_list = df[col].tolist()
		for ii in range(0, len(col_list)):
			if col_list[ii] in mapping['stop']:
				newcol.append('stop')
			elif col_list[ii] in mapping['fricative']:
				newcol.append('fricative')
			elif col_list[ii] in mapping['vowel']:
				newcol.append('vowel')
			elif col_list[ii] in mapping['nasal']:
				newcol.append('nasal')
			elif col_list[ii] in mapping['approximant']:
				newcol.append('approximant')
			elif col_list[ii] in mapping['eow']:
				newcol.append('eow')
			else:
				print(col_list[ii])
				print("Could not find this segment in nsdict")
		df[col + "_SegClass"] = newcol


def analyzeTailContexts(ems_by_cpair, longprop_by_context, allcontexts):
	ranks = getFrequencyRank(allcontexts)
	# print(allcontexts)
	# print(ranks[(1, 1, 1, 1, 'e', '#', '#')])
	# print(ranks[(0, 0, 0, 0, 'e', '#', '#')])
	# print(ranks[(1, 0, 0, 0, 'e', '#', '#')])
	# print(ranks[(1, 0, 1, 1, 'e', '#', '#')])
	# print(ranks[(0, 0, 1, 1, 'e', '#', '#')])
	# print(ranks[(1, 0, 1, 1, 'e', '#', '#')])
	# print(ranks[(1, 0, 1, 1, 'e', '#', '#')])

	for k, v in ems_by_cpair.items():
		if v < 0.225:
			continue
		# elif longprop_by_context[k[0]] < 0.1 and longprop_by_context[k[1]] < 0.1:
		# 	continue
		# elif ranks[k[0]] > 50 and ranks[k[1]] > 50:
		# 	continue

		else:
			print(str(v))
			print('-'*50)
			print("Context 1: " + str(k[0]))
			print("Percent Long: " + str(longprop_by_context[k[0]]))
			print("Frequency: " + str(allcontexts[k[0]]))
			print("Rank: " + str(ranks[k[0]]))

			print("Context 2: " + str(k[1]))
			print("Percent Long: " + str(longprop_by_context[k[1]]))
			print("Frequency: " + str(allcontexts[k[1]]))
			print("Rank: " + str(ranks[k[1]]))
			print('-'*50)

# recodeDur0to1 takes a pandas dataframe (inc. duration + context),
# and adds a new column Duration_0to1.
# Duration_0to1 rescales all of the durations to be between 0 and 1 
def recodeDur0to1(df):
	minDur = min(df['Duration'])
	maxDur = max(df['Duration'])
	df['Duration_0to1'] = (df['Duration'] - minDur)/(maxDur - minDur)
	return(df)

# zscoreDur takes a pandas dataframe (incl. duration + context),
# and adds a new column Duration_z
# Duration_z zcores all of the durations
def zscoreDur(df):
	df['Duration_z'] = zscore(df['Duration'])
	return(df)

# def makeDensityPlots():

def roundDuration(df):
	df.round({'Duration':2})
	return(df)


# getContexts gets every context, its frequency, and the durations within it from 
# a list (basically a dataframe)
def getContexts(df, header, durtype, c):

	# Get index of all of the important contextual information
	idx_pi = header.index("UttInitial")
	idx_pf = header.index("UttFinal")
	idx_wi = header.index("WordInitial")
	idx_wf = header.index("WordFinal")
	idx_noisypi = header.index("UttInitial_Noisy")
	idx_noisypf = header.index("UttFinal_Noisy")
	idx_noisywi = header.index("WordInitial_Noisy")
	idx_noisywf = header.index("WordFinal_Noisy")
	idx_qual = header.index("Quality")
	idx_dur = header.index(durtype)
	idx_prevseg = header.index("ModPrevSeg")
	idx_follseg = header.index("ModFollSeg")
	idx_prevsegclass = header.index("ModPrevSeg_SegClass")
	idx_follsegclass = header.index("ModFollSeg_SegClass")
	idx_len = header.index("Length")
	idx_spkr = header.index("Speaker")
	idx_word = header.index("WordFrame")

	# Dict to keep track of contexts (keys) and their frequencies (values)
	allcontexts = {}
	# Dict to keep track of contexts (keys) and a list of durations in that context (values)
	durs_by_context = defaultdict(list)
	# Dict to keep track of contexts (keys) and their short vowel proportions
	longtotal_by_context = {}

	# Loop through the data frame
	for vowel in df:
		# Get context in a tuple
		# Case 1: Context under consideration = wordframes
		if c == 'wf':
			context_of_this_vowel = tuple([vowel[idx_word]])
		# Case 2: Context under consideration = prosody + neighboring sounds + quality
		elif c == 'vq':
			context_of_this_vowel = tuple([vowel[idx_qual]])
		elif c == 'ns':
			context_of_this_vowel = (vowel[idx_prevseg], vowel[idx_follseg])
		elif c == 'nsclass':
			context_of_this_vowel = (vowel[idx_prevsegclass], vowel[idx_follsegclass])
		elif c == 'p':
			context_of_this_vowel = (vowel[idx_pi], vowel[idx_pf], vowel[idx_wi], vowel[idx_wf])
		elif c == 'combonsclass':
			context_of_this_vowel = (vowel[idx_pi], vowel[idx_pf], vowel[idx_wi], vowel[idx_wf], vowel[idx_qual], vowel[idx_prevsegclass], vowel[idx_follsegclass])
		elif c == 'noisycombonsclass':
			context_of_this_vowel = (vowel[idx_noisypi], vowel[idx_noisypf], vowel[idx_noisywi], vowel[idx_noisywf], vowel[idx_qual], vowel[idx_prevsegclass], vowel[idx_follsegclass])
		else:
			context_of_this_vowel = (vowel[idx_pi], vowel[idx_pf], vowel[idx_wi], vowel[idx_wf], vowel[idx_qual], vowel[idx_prevseg], vowel[idx_follseg])
		
		# Add context to allcontexts, update frequency, update lst of durations
		if context_of_this_vowel not in allcontexts:
			allcontexts[context_of_this_vowel] = 1.
			durs_by_context[context_of_this_vowel] = [float(vowel[idx_dur])]
		else:
			allcontexts[context_of_this_vowel] += 1.
			durs_by_context[context_of_this_vowel].append(float(vowel[idx_dur]))

		# Add context to longtotal_by_context
		if context_of_this_vowel not in longtotal_by_context:
			longtotal_by_context[context_of_this_vowel] = int(vowel[idx_len])
		else:
			longtotal_by_context[context_of_this_vowel] += int(vowel[idx_len])

	return(allcontexts, durs_by_context, longtotal_by_context)

# def getContextualInfo(full_sl, full_so, used_sl, used_so, freq_sl, freq_so):
# 	# print(freq_sl)
# 	# df = {'Contrastive': used_sl, 'Non-contrastive': used_so}
# 	# df = pd.DataFrame(df) 
# 	# df[["Contrastive", "Non-contrastive"]].plot.kde()
# 	a = sns.distplot(freq_sl, hist=True, rug=False, label = 'Contrastive')
# 	sns.distplot(freq_so, hist=True, rug=False, label = 'Non-contrastive')
# 	title1 = "Contrastive = " + str(len(freq_sl)) + " contexts; Non-contrastive = " + str(len(freq_so)) + " contexts\n"
# 	title2 = "Contrastive: Minimum Frequency: " + str(min(freq_sl)) + " Maximum Frequency: " + str(max(freq_sl)) + "\n"
# 	title3 = "Non-contrastive: Minimum Frequency: " + str(min(freq_so)) + " Maximum Frequency: " + str(max(freq_so)) + "\n"
# 	title = title1 + title2 + title3
# 	a.set_title(title)
# 	a.set_ylabel('Density')
# 	a.set_xlabel('Number of Tokens in Context')
# 	plt.show()
	
# 	print("Used contrastive: " + str(len(freq_sl)) + " contexts")
# 	print("Used non-contrastive: " + str(len(freq_so)) + " contexts")
# 	print("Total contrastive: " + str(len(full_sl)) + " contexts")
# 	print("Total non-contrastive: " + str(len(full_so)) + " contexts")
# 	print("Max contrastive frequency: " + str(max(freq_sl)))
# 	print("Min contrastive frequency: " + str(min(freq_sl)))
# 	print("Max non-contrastive frequency: " + str(max(freq_so)))
# 	print("Min non-contrastive frequency: " + str(min(freq_so)))
	

# getSubsetofContexts takes a dictionary
def getSubsetofContexts(allcontexts_sl, allcontexts_so, cutofftype, cutoffval):

	# List of tuples (FREQUENCY, CONTEXT)
	extractfrequencies_sl = []
	extractfrequencies_so = []

	# List of frequencies with no associated context
	frequencies_sl = []
	frequencies_so = []

	if cutofftype == 'rank':
		cutoffval = min(len(allcontexts_sl), len(allcontexts_so), cutoffval)

	# Get frequencies associated with each context
	for item in allcontexts_sl:
		frequencies_sl.append(allcontexts_sl[item])
		extractfrequencies_sl.append((allcontexts_sl[item], item))
	for item in allcontexts_so:
		frequencies_so.append(allcontexts_so[item])
		extractfrequencies_so.append((allcontexts_so[item], item))

	# Sort these lists from most frequent to least frequent
	frequencies_sl.sort(reverse = True)
	frequencies_so.sort(reverse = True)
	extractfrequencies_sl.sort(key=lambda x: x[0], reverse = True)
	extractfrequencies_so.sort(key=lambda x: x[0], reverse = True)

	# print(extractfrequencies_so)
	# print(extractfrequencies_sl)
	# print(len(extractfrequencies_sl))
	# print(len(extractfrequencies_so))

	# print(extractfrequencies_sl[0:cutoffval])
	# print(extractfrequencies_so[0:cutoffval])

	# Subset of contexts that we will further analyze
	contexts_sl = []
	contexts_so = []
	freqs_sl = []
	freqs_so = []

	# We are cutting off based on the number of sounds that occur in the context
	if cutofftype == 'freq':
		# Loop through every context and check if its frequency exceeds the cut off
		# If so, add it to the list. Repeat for both sl and so.
		for item in allcontexts_sl:
			if allcontexts_sl[item] >= cutoffval:
				contexts_sl.append(item)
				freqs_sl.append(allcontexts_sl[item])

		for item in allcontexts_so:
			if allcontexts_so[item] >= cutoffval:
				contexts_so.append(item)
				freqs_so.append(allcontexts_so[item])

		# print("Included contexts for contrastive case:")
		# for ii in range(0, len(extractfrequencies_sl)):
		# 	if extractfrequencies_sl[ii][0] >= cutoffval:
		# 		print(extractfrequencies_sl[ii])

		# print("Included contexts for non-contrastive case:")
		# for ii in range(0, len(extractfrequencies_so)):
		# 	if extractfrequencies_so[ii][0] >= cutoffval:
		# 		print(extractfrequencies_so[ii])

	# We are cutting off based on the rank of the context (by frequency)
	elif cutofftype == 'rank':
		# Loop through every context and check if its rank exceeds the cut off
		# If so, add it to the list. Repeat for both sl and so.
		# This is determined by figuring out the frequency of the lowest rank allowable
		# and checking if the current context has higher frequency than that
		for item in allcontexts_sl:
			if allcontexts_sl[item] >= frequencies_sl[cutoffval-1]:
				contexts_sl.append(item)
				freqs_sl.append(allcontexts_sl[item])

		for item in allcontexts_so:
			if allcontexts_so[item] >= frequencies_so[cutoffval-1]:
				contexts_so.append(item)
				freqs_so.append(allcontexts_so[item])

		contexts_sl = contexts_sl[0:cutoffval]
		freqs_sl = freqs_sl[0:cutoffval]
		contexts_so = contexts_so[0:cutoffval]
		freqs_so = freqs_so[0:cutoffval]

	# getContextualInfo(allcontexts_sl, allcontexts_so, contexts_sl, contexts_so, freqs_sl, freqs_so)
	# wkeljrlwejrlkwj
		# print("Included contexts for contrastive case:")
		# print(extractfrequencies_sl[0:cutoffval])
		# print("Included contexts for non-contrastive case:")
		# print(extractfrequencies_so[0:cutoffval])

	return(contexts_sl, contexts_so)

# Take a list of contexts and dictionary with their duration values and get a list of EMs
def getEMs(contexts, durs_by_context, measure, durationrange):
	ems = []
	ems_by_contextpair = {}
	for ii in range(0, len(contexts)):
		for jj in range(ii+1, len(contexts)):
			c1_dur = durs_by_context[contexts[ii]]
			c2_dur = durs_by_context[contexts[jj]]

			# Subtract mean
			# print("SHIFTING CONTEXT MEANS")
			# c1_dur = np.array(c1_dur) - np.mean(np.array(c1_dur))
			# c2_dur = np.array(c2_dur) - np.mean(np.array(c2_dur))

			# print("SKIPPING SOME CONTEXTS")
			# if contexts[ii] == ('#Nd_e_',) or contexts[jj] == ('#Nd_e_',):
			# 	continue

			# contexts_in_question = [(1, 0, 1, 1, 'e', '#', '#'), (0, 1, 1, 1, 'e', '#', '#'), (0, 0, 1, 1, 'e', '#', '#'), (0, 0, 0, 1, 'e', 'n', '#')]
			# contexts_in_question = [('#_e_',)]
			# contexts_in_question = [('#h_e_',)]

			# if contexts[ii] in contexts_in_question or contexts[jj] in contexts_in_question:
			# 	continue
			
			if measure == 'em':
				em = wasserstein_distance(c1_dur, c2_dur)
			elif measure == 'kl':
				em = symmetric_kl_divergence(c1_dur, c2_dur, durationrange, n_bins=20)
			elif measure == 'js':
				em = js_divergence(c1_dur, c2_dur, n_bins=10)
			elif measure == 'hellinger':
				em = hellinger_distance(c1_dur, c2_dur, n_bins=10) 
			else:
				print("Unrecognized measure")
			ems.append(em)
			ems_by_contextpair[(contexts[ii], contexts[jj])] = em

	return(ems, ems_by_contextpair)

# makeBoxPlot takes as input a list of EMs for the contrastive and non-contrastive cases under
# conmparison. sl_label and so_label are the labels that go on the output boxplot
def makeBoxPlot(em_sl, em_so, sl_label, so_label, boxplotfilename, show):
	testem = [em_so, em_sl]
	plt.boxplot(testem, labels = [so_label+'\n(no contrast)', sl_label+'\n(contrast)'], widths=0.3)
	plt.ylabel("Earthmover's Distance")#, #fontsize = 20)
	# plt.xticks(fontsize = 20)
	# plt.yticks(fontsize = 20)
	# plt.axes().set_aspect(0.005)
	# plt.axes().set_aspect(0.005)
	# plt.ylim((-1,5))
	# plt.yticks([-1,0,1,2,3,4,5], fontsize = 20)
	fig1 = plt.gcf()
	if show == True:
		plt.show(block=False)
	fig1.savefig(boxplotfilename, bbox_inches = 'tight')
	# plt.close()

def getLongProp(longtot_by_context, allcontexts):
	longprop_by_context = {}
	for context in allcontexts:
		longprop_by_context[context] = longtot_by_context[context] / allcontexts[context]
	return(longprop_by_context)


def DLbycontext(sl, so, header, col, context, freqrank, freqrankval, measure, durationrange):
	# allcontexts = dictionary with keys CONTEXT and values FREQUENCY
	# durs_by_context = dictionary with keys CONTEXT and values LST_OF_DURS
	# long_by_context = dictionary with keys CONTEXT and values # of long vowels in that context
	# Get all of the contexts (and their freqs) in each of the sl and so dataframes
	# as well as a list of every duration corresponding to a particular context
	allcontexts_sl, durs_by_context_sl, longtot_by_context_sl = getContexts(sl,header,col,context)
	allcontexts_so, durs_by_context_so, longtot_by_context_so = getContexts(so,header,col,context)
	print("SL num of contexts: " + str(len(allcontexts_sl)))
	print("SO num of contexts: " + str(len(allcontexts_so)))

	# Get proportion of long vowels in each context
	longprop_by_context_sl = getLongProp(longtot_by_context_sl, allcontexts_sl)
	# print("Length of all contexts - contrast")
	# print(len(allcontexts_sl))
	# print("Length of all contexts - no contrast")
	# print(len(allcontexts_so))

	# Get a list of the subset of contexts to consider in further analyses
	# freq = The threshold for context inclusion is based on its frequency (e.g. contexts
	# that have more than 100 sounds in them)
	# rank = The threshold for context inclusion is based on its rank (e.g. the top 10 contexts)
	# In the case of 'freq', the following number is that frequency threshold (>100 = 100)
	# In the case of 'rank', the following number is that rank (top 100 = 100)
	# contexts_sl, contexts_so = getSubsetofContexts(allcontexts_sl, allcontexts_so, freqrank, freqrankval)
	# Returns a list
	contexts_sl, contexts_so = getSubsetofContexts(allcontexts_sl, allcontexts_so, freqrank, freqrankval)
	# print(len(contexts_sl))
	# REMOVED FROM WF
	# contexts_sl.remove(('#_e_',))
	# print(len(contexts_sl))
	# REMOVED FROM COMBO
	# contexts_sl.remove((1, 0, 1, 1, 'e', '#', '#'))
	# contexts_sl.remove((0, 1, 1, 1, 'e', '#', '#'))
	# contexts_sl.remove((0, 0, 1, 1, 'e', '#', '#'))
	# contexts_sl.remove((0, 0, 0, 1, 'e', 'n', '#'))

	# REMOVED FROM WF DUTCH IDS
	# contexts_sl.remove(('#h_e_',))
	# contexts_sl.remove(('#j_a_',))
	# print("Length of used contexts - contrast")
	# print(len(contexts_sl))
	# print("Length of used contexts - no contrast")
	# print(len(contexts_so))

	# Loop through all pairs of contexts and get EM between them (one list for each dataset)
	em_sl, ems_by_cpair_sl = getEMs(contexts_sl, durs_by_context_sl, measure, durationrange)
	em_so, ems_by_cpair_so = getEMs(contexts_so, durs_by_context_so, measure, durationrange)

	# TODO TODO
	# ems.append((contexts[ii],contexts[jj],em))

	# Returns lists)
	return(em_sl, em_so, ems_by_cpair_sl, ems_by_cpair_so, longprop_by_context_sl, allcontexts_sl)

# generateBootstripSample samples with replacement a new set of rows of durations + contexts
# data could be e.g. French or Japanese data
# nSamples is # of rows to include in the new data
def generateBootstrapSample(data, nSamples):
	newdata = []
	for ii in range(0, nSamples):
		# Randomly pick a row (with replacement)
		rownum = randrange(len(data))
		newdata.append(data[rownum])
	return(newdata)

# Read in sl (shortlong) and so (short only) dataframes
# and z-score

def setLanguage(str):
	if str == 'JapaneseFrench':
		sldirectory = './CSJ/'
		sodirectory = './NCCFr/'
		slfile = sldirectory + 'csj.csv'
		sofile = sodirectory + 'nccfr.csv'
		sllabel = 'Japanese ADS'
		solabel = 'French ADS'
	elif str == 'DutchADS':
		sldirectory = './ECSD/'
		sodirectory = './ECSD/'
		slfile = sldirectory + 'ecsd_shortlong.csv'
		sofile = sodirectory + 'ecsd_shortonly.csv'
		sllabel = 'Dutch ADS'
		solabel = 'Dutch ADS'
	else:
		sldirectory = './DutchIDS/'
		sodirectory = './DutchIDS/'
		slfile = sldirectory + 'dutchids_shortlong.csv'
		sofile = sodirectory + 'dutchids_shortonly.csv'
		sllabel = 'Dutch IDS'
		solabel = 'Dutch IDS'
	return(sldirectory, sodirectory, slfile, sofile, sllabel, solabel)


# # French vs. Japanese
# sldirectory = './CSJ/'
# sodirectory = './NCCFr/'
# slfile = sldirectory + 'csj.csv'
# sofile = sodirectory + 'nccfr.csv'
# sllabel = 'Japanese ADS'
# solabel = 'French ADS'

## slfile = './CSJ/csj.csv'
## sofile = './NCCFr/nccfr.csv'


# sldirectory = './ECSD/'
# sodirectory = './ECSD/'
# slfile = sldirectory + 'ecsd_shortlong.csv'
# sofile = sodirectory + 'ecsd_shortonly.csv'
# sllabel = 'Dutch ADS'
# solabel = 'Dutch ADS'

## slfile = './ECSD/ecsd_shortlong.csv'
## sofile = './ECSD/ecsd_shortonly.csv'

# sldirectory = './DutchIDS/'
# sodirectory = './DutchIDS/'
# slfile = sldirectory + 'dutchids_shortlong.csv'
# sofile = sodirectory + 'dutchids_shortonly.csv'
# sllabel = 'Dutch IDS'
# solabel = 'Dutch IDS'

# slfile = './DutchIDS/dutchids_shortlong.csv'
# sofile = './DutchIDS/dutchids_shortonly.csv'

# SET LANGUAGE
# CHANGE LANGUAGE
# lang = 'DutchADS'
# lang = 'DutchIDS'
lang = 'JapaneseFrench'

# SET CONTEXT TYPE
# contexttype = 'vq'	# vowel quality
# contexttype = 'p'		# prosody
# contexttype = 'ns'	# neighboring sound
contexttype = 'wf'	# word frame
# contexttype = 'combo'	# p + ns + vq
# contexttype = 'combonsclass'	# p + ns (defined as broad classes) + vq
# contexttype = 'nsclass'		# ns (defined as broad classes)
# contexttype = 'noisycombonsclass'		# noise added to p + ns (broad class) + vq

# SET DURATION TYPE
# cuetype = 'Duration_z'
cuetype = 'Duration'

# SET FREQ/RANK
# freqrank = 'freq'
freqrank = 'rank'

# SET FREQ/RANK VALUE
# freqrankval = 200
# freqrankval = 500
# freqrankval = 5
freqrankval = 200

# SET NUMBER OF ITERATIONS
nIter = 1

# measure = 'em'
measure = 'kl'
# measure = 'js'
# measure = 'hellinger'

# noise = True
# soundclass = True

# SET NUMBER OF SAMPLES

all_slvar = []
all_sovar = []
all_slmean = []
all_somean = []
all_slmax = []
all_somax = []
all_slrange = []
all_sorange = []
all_nSamples = []
all_nContexts = []

# Dutch ADS
# nSamplesList = [284, 500, 1000, 2000, 5000, 10000, 21187]
# Japanese vs French
# nSamplesList = [284, 500, 1000, 2000, 5000, 10000, 21187, 50000, 132037]
# nSamplesList = [5000, 10000]
# Dutch IDS
# nSamplesList = [284]
# nSamplesList = [132037]
if lang == 'JapaneseFrench':
	nSamplesList = [132037]
elif lang == 'DutchADS':
	nSamplesList = [21187]
elif lang == 'DutchIDS':
	nSamplesList = [284]
else:
	print("Unrecognized language")

# nSamples2nContexts = {284: 200, 500: 200, 1000: 200, 2000: 200, 5000: 200, 10000: 200, 21187: 200, 50000: 200, 132037: 200}
# nSamples2nContexts = {284: 5, 500: 5, 1000: 10, 2000: 20, 5000: 50, 10000: 100, 21187: 200, 50000: 200, 132037: 200}
nSamples2nContexts = {132037: 200, 21187: 200, 284: 5}


for N in nSamplesList:

	sldirectory, sodirectory, slfile, sofile, sllabel, solabel = setLanguage(lang)
	# print('1')
	sl = pd.read_csv(slfile)
	so = pd.read_csv(sofile)

	# print(len(sl['Length']))
	# print(len(so['Length']))
	# wekjrwlkejrj
	# print("GETTING RID OF LONG VOWELS")
	# sl = sl[sl['Length'] < 1]
	# print(len(sl['Length']))
	# wekjrwklejr
	# pd.set_option('display.max_rows', None)
	# print(sl)
	# wekjwlkerjlkwe

	# roundDuration(sl)
	# roundDuration(so)

	sl = sl.round({'Duration':2})
	so = so.round({'Duration':2})

	# TODO: added 12/7
	min_duration = min(min(sl['Duration'].tolist()), min(so['Duration'].tolist()))
	max_duration = max(max(sl['Duration'].tolist()), max(so['Duration'].tolist()))
	durationrange = (min_duration, max_duration)
	# print("Min duration is " + str(min_duration))
	# print("Max duration is " + str(max_duration))

	# print(sl.Duration)
	# print(so.Duration)
	# wejlrwer

	# print("PERMUTING VOWEL ORDER")
	# sl['Duration'] = np.random.permutation(sl['Duration'].values)
	# so['Duration'] = np.random.permutation(so['Duration'].values)

	# print(sl['Duration'].tolist())
	# wkjelrwlek

	# for col in sl.columns:
	# 	print(col)
	# wlekjrlwekj
	recodeDur0to1(sl)
	recodeDur0to1(so)

	zscoreDur(sl)
	zscoreDur(so)

	# # TODO: deal with
	# sl_prevseg = sl['PrevSeg'].tolist()
	# sl_prevseg = np.unique(sl_prevseg).tolist()
	# # print(sl_prevseg)

	# so_prevseg = so['PrevSeg'].tolist()
	# so_prevseg = np.unique(so_prevseg).tolist()
	# # print(sl_prevseg + so_prevseg)

	# sl_follseg = sl['FollSeg'].tolist()
	# sl_follseg = np.unique(sl_follseg).tolist()
	# # print(sl_follseg)

	# so_follseg = so['FollSeg'].tolist()
	# so_follseg = np.unique(so_follseg).tolist()
	# # print(sl_follseg + so_follseg)

	# print(np.unique(so_prevseg + so_follseg))

	# wkerljweltkjwelktjwe

	mapping_sl, mapping_so = seg2class(lang)
	nsClass(sl, mapping_sl)
	nsClass(so, mapping_so)

	# print(sl.head())
	# # so.head()
	# weljrwker

	# print("Adding contextual noise")
	addContextualNoise(sl, ['WordInitial', 'WordFinal', 'UttInitial', 'UttFinal'], 0.2)
	fixProsody(sl)
	addContextualNoise(so, ['WordInitial', 'WordFinal', 'UttInitial', 'UttFinal'], 0.2)
	fixProsody(so)
	# addContextualNoise(so, ['UttInitial', 'UttFinal'], 0.2)


	header = list(sl)
	# print(header)
	# wlkjerlkwejrl
	# print(header)
	# wlkejrwekljr
	sl = sl.values.tolist()
	so = so.values.tolist()

	# totalCorpus is the length of the shorter dataset
	totalCorpus = min(len(sl), len(so))
	# Set nSamples to the minimum of N and the size of the datasets
	nSamples = min(N, totalCorpus)

	# print(totalCorpus)
	# print(nSamples)


	# Set freqrankval proportionally, but the minimum is 5
	freqrankval = nSamples2nContexts[N]
	# freqrankval = max(5, round((500./132307)*nSamples))
	# freqrankval = max(5, round((200./132307)*nSamples))
	# freqrankval = min(totalCorpus, freqrankval)

	# print(freqrankval)
	# wkejrwer

	# print("SHUFFLING DATA")
	# random.shuffle(sl)
	# random.shuffle(so)

	# sl = sl[0:nSamples]
	# so = so[0:nSamples]
	# print('5')

	# Subset the corpora to the same size
	sl = sl[0:totalCorpus]
	so = so[0:totalCorpus]

	slvar = []
	sovar = []
	slmean = []
	somean = []
	slmax = []
	somax = []
	slrange = []
	sorange = []


	sl = sl[0:nSamples]
	so = so[0:nSamples]

	# if nIter == 1:
	# 	em_sl, em_so, ems_by_cpair_sl, ems_by_cpair_so, longprop_by_c_sl, allcontexts_sl = DLbycontext(sl, so, header, cuetype, contexttype, freqrank, freqrankval, measure)
	
	for ii in range(0, nIter):
		print(nSamples, freqrank, freqrankval, ii)

		if nIter > 1:
			slsample = generateBootstrapSample(sl, nSamples)
			sosample = generateBootstrapSample(so, nSamples)
		else:
			slsample = sl
			sosample = so

		em_sl_sample, em_so_sample = DLbycontext(slsample, sosample, header, cuetype, contexttype, freqrank, freqrankval, measure, durationrange)[0:2]
		all_slvar.append(np.var(em_sl_sample))
		all_sovar.append(np.var(em_so_sample))
		all_slmean.append(np.mean(em_sl_sample))
		all_somean.append(np.mean(em_so_sample))
		all_slmax.append(max(em_sl_sample))
		all_somax.append(max(em_so_sample))
		all_slrange.append(max(em_sl_sample) - min(em_sl_sample))
		all_sorange.append(max(em_so_sample) - min(em_so_sample))
		all_nSamples.append(nSamples)
		all_nContexts.append(freqrankval)

		slvar.append(np.var(em_sl_sample))
		sovar.append(np.var(em_so_sample))
		slmean.append(np.mean(em_sl_sample))
		somean.append(np.mean(em_so_sample))
		slmax.append(max(em_sl_sample))
		somax.append(max(em_so_sample))
		slrange.append(max(em_sl_sample) - min(em_sl_sample))
		sorange.append(max(em_so_sample) - min(em_so_sample))


# Output summary results to file (this is used for the line plots in the paper)
filename = contexttype + '_' + cuetype + '_' + 'nIter=' + str(nIter) + '_' + freqrank + '=' + str(freqrankval) + '_' + measure + '_summary.csv' 
sllst = ['contrastive']*nIter*len(nSamplesList)
solst = ['non-contrastive']*nIter*len(nSamplesList)
# headerout = ['ContrastType', 'Variance', 'Mean', 'Maximum', 'Range']
data = {'ContrastType': sllst+solst, 'InputSize':all_nSamples + all_nSamples, 'FreqRankVal': all_nContexts + all_nContexts, 'Variance':all_slvar + all_sovar, 'Mean':all_slmean + all_somean, 'Maximum': all_slmax + all_somax, 'Range': all_slrange + all_sorange}
df = pd.DataFrame(data) 
df.to_csv(sldirectory + 'Results/' + filename, index=False)

# If running one iteration, this means we're using the raw data and not bootstrap samples
# Output full information to file
if nIter == 1:
	filename_sl = filename.replace('_summary.csv', '_contrast.csv')
	filename_so = filename.replace('_summary.csv', '_nocontrast.csv')
	data_sl = {'EM': em_sl_sample, 'Status': ['contrastive']*len(em_sl_sample)}
	df_sl = pd.DataFrame(data_sl)
	df_sl.to_csv(sldirectory + 'Results/' + filename_sl, index = False)
	data_so = {'EM': em_so_sample, 'Status': ['non-contrastive']*len(em_so_sample)}
	df_so = pd.DataFrame(data_so)
	df_so.to_csv(sldirectory + 'Results/' + filename_so, index = False)




# def makeLinePlot(df, depmeasure, filename, outdir):
# 	ax = sns.pointplot(x='InputSize', y=depmeasure, hue='ContrastType', data=df)
# 	ax.figure.savefig(outdir + filename + '_' + depmeasure + ".png")
# 	plt.show()

# # makeLinePlot(df, 'Variance', filename, )
# depmeasure = 'Variance'
# ax = sns.pointplot(x='InputSize', y=depmeasure, hue='ContrastType', data=df)
# filename = contexttype + '_' + cuetype + '_' + 'nIter=' + str(nIter) + '_' + freqrank + '=' + str(freqrankval) + '_inputsize.csv' 
# ax.figure.savefig(outdir + a + '_' + depmeasure + ".png")
# plt.show()


