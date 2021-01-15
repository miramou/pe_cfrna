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