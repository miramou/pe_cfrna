## Plot specific defaults across figures

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from nheatmap import nhm, scripts

#Set fig export settings

#For editable text. Except latex text is still shapes sadly
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sns.set_style("whitegrid")
font = {'size' : 6}
lines = {'linewidth' : 0.5}
fig = {'figsize' : (2.5, 1.5)}

mpl.rc('font', **font)
mpl.rc('lines', **lines)
mpl.rc('figure', **fig)

#Set style
sns.set(style="whitegrid", palette="pastel", color_codes=True)

def make_color_map(name, color_list):
    return mpl.colors.LinearSegmentedColormap.from_list(name, color_list, len(color_list))
    
#Set color maps
outlier_palette = ["#018571", "#dfc27d"] #non-out, outlier
outlier_palette2 = ["#018571", "#d8b365", "#8c510a"] #non-out, outlier, nc

discovery_validation_palette = ['#1f78b4', '#9ecae1'] #discovery, val
discovery_test1_palette = ['#1f78b4', '#33a02c', '#33a02c'] #discovery, test
discovery_test2_palette = ['#1f78b4', '#1f78b4', '#33a02c', '#b2df8a'] #discovery, test
discovery_munchel_palette = ['#1f78b4', '#b2df8a', '#b2df8a'] #discovery, test, test
cntrl_pe_palette = ["#92c5de", "#ca0020"]
cntrl_pe_pe_w_cond_palette = ["#92c5de", "#f4a582", "#ca0020"]
pe_feat_palette = ["#f7f7f7", "#b2182b", "#67001f"] # cntrl, mild, severe
pe_feat_palette2 = ["#000000", "#e08214", "#b2182b"] # cntrl, mild, severe
pe_type_palette = ["#f7f7f7", "#67001f", "#b2182b"] # cntrl, early, late

#Set term label mapping
term_labels = {1 : '<= 12 weeks gestation', 2 : '13-20 weeks gestation', 3 : '>= 23 weeks gestation', 4 : 'Post-partum'}