## Plot specific utilities

from util.plot_default import *
import pandas as pd

def save_figure_pdf(fig, path_to_save):
	'''
	Utility fxn to save a figure without having to retype all the required options.
	Input:
		fig - a matplotlib.pyplot.figure instance with the figure you'd like to save
		path_to_save - the path you'd like to save the figure to

	Returns: Nothing but there should now be a pdf version in 300 dpi with a transparent bkg at path_to_save
	'''
	fig.savefig(path_to_save, dpi = 300, bbox_inches = 'tight', transparent = True)
	
def DE_plot_order(DE_series, key):
	'''
	Utility function defining the category order for DE logFCs called in figure 2D
	Input: 
		DE_series - An unordered series containing values that match either category key
		key - Accepts 'kmeans' as value
	Return: Ordered pd categories
	'''
	if key == 'kmeans':
		categories = ['1', '2', '3']
	return pd.Categorical(DE_series, ordered = True, categories=categories)

def nhm_plot_heatmap(to_plot, dfr = None, dfc = None, cmaps = None, cmapCenter_color = 'RdBu_r', **kwargs):
	'''
	Utility function to plot heatmap + hierarchical clustering
	Input: 
		to_plot - pd df where rows and columns correspond to those for heatmap
		dfr - pd df with labels for *rows* of heatmap to visualize with color bar
		dfc - pd df with labels for *columns* of heatmap to visualize with color bar
		cmaps - dictionary of color maps where key corresponds to column name in dfc or dfr and value corresponds to color map to use
		**kwargs - other arguments to pass to g.run() - see nheatmap for details
	Return: figure and plot as returned by nheatmap fxn g.run()
	'''
	if cmaps is None:
		g = nhm(data=to_plot, dfr = dfr, dfc=dfc, figsize=(10, 10), linewidths=0, cmapCenter=cmapCenter_color, 
		xrot = 90, wspace = 0.05, showxticks = False) 
	else:
		g = nhm(data=to_plot, dfr = dfr, dfc=dfc, figsize=(10, 10), linewidths=0, cmapCenter=cmapCenter_color, 
			cmaps = cmaps, xrot = 90, wspace = 0.05, showxticks = False) 
	g.hcluster(method='average', metric='euclidean', optimal_ordering=True)
	fig, plot = g.run(**kwargs)
	return fig, plot

def plot_roc_curves(roc_auc_dict, linestyles, palette, save_path):
	fig, ax = plt.subplots()

	plot_i = 0
	for label, roc_auc in roc_auc_dict.items():
		plt.plot(roc_auc['fpr'], roc_auc['tpr'], color = discovery_test_palette[plot_i], lw = 1.5, linestyle = linestyles[plot_i],
				 label = '%s (AUC = %.2f [%.2f - %.2f])' % (label, roc_auc['auc'], roc_auc['ci_auc_lb'], roc_auc['ci_auc_ub']))
		plot_i += 1

	plt.plot([0, 1], [0, 1], color='gray', linestyle='--') #No discrimination line
	plt.xlabel("False positive rate")
	plt.ylabel("True positive rate")

	plt.legend(loc = 'lower right')
	save_figure_pdf(fig, save_path)
	return


def plot_violin_and_swarmplot(df, x_name, y_name, hue_name, orient, violinplot_palette, x_label, y_label, split = False, inner = 'sticks', figsize = (5, 5),
							include_vline = False, include_hline = False, line_pos = None, line_start = None, line_end = None):
	'''
	Plotting utility to overlay swarmplot and violinplot

	Inputs:
		df - dataframe used for plotting
		x_name - x var column name in df (str)
		y_name - y var column name in df (str)
		hue_name - hue var column name in df (str)
		orient - either 'h' or 'v', see seaborn (str)
		violinplot_palette - palette used for violinplot hue (list)
		x_label - x axis label (str)
		y_label - y axis label (str)
		split - split violinplot (bool)
		inner - violinplot inner (str) see seaborn
		figsize - figure size (tuple)
		include_vline - whether to include a vertical line, usually either this or hline (bool)
		include_hline - whether to include a horizontal line, either this or vline (bool)
		line_pos - the fixed location to plot line (x axis for vline or y axis hline depending on bool) (int)
		line_start - the starting location to plot line (y axis for vline or x axis for hline depending on bool) (int)
		line_end - the ending location to plot line (y axis for vline or x axis for hline depending on bool) (int)

	Returns:
		fig, ax handles for plot
	'''
	fig, ax  = plt.subplots(1, figsize = figsize)

	if include_vline:
		plt.vlines(line_pos, line_start, line_end, linestyle = '--', lw = 2, color = 'gray')
	if include_hline:
		plt.hlines(line_pos, line_start, line_end, linestyle = '--', lw = 2, color = 'gray')

	sns.swarmplot(x = x_name, y = y_name, hue = hue_name, orient = orient, dodge = split, data = df,
					size = 6, lw = 2, edgecolor = 'k', color = 'k')
	
	sns.violinplot(x = x_name, y = y_name, hue = hue_name, orient = orient, split = split, dodge = False, inner = inner, cut = 0,
					data = df, palette = violinplot_palette)

	plt.xlabel(x_label)
	plt.ylabel(y_label)

	return fig, ax
