from util.plot_default import *
import pandas as pd

def DE_plot_order(DE_series, key):
	#A utility function defining the category order for DE logFCs called for figures 2D/E
	categories = ['Earlygestation_down', 'Earlygestation_up', 'Mid-gestation_down', 'Mid-gestation_up', 
				'Lategestation_down', 'Lategestation_up', 'Post-partum_down', 'Post-partum_up']
	if key == 'kmeans':
		categories = ['1', '2', '3']
	return pd.Categorical(DE_series, ordered = True, categories=categories)

def nhm_plot_heatmap(to_plot, dfr = None, dfc = None, cmaps = None, **kwargs):
	if cmaps is None:
		g = nhm(data=to_plot, dfr = dfr, dfc=dfc, figsize=(10, 10), linewidths=0, cmapCenter="RdBu_r", 
		xrot = 90, wspace = 0.05, showxticks = False) 
	else:
		g = nhm(data=to_plot, dfr = dfr, dfc=dfc, figsize=(10, 10), linewidths=0, cmapCenter="RdBu_r", 
			cmaps = cmaps, xrot = 90, wspace = 0.05, showxticks = False) 
	g.hcluster(method='average', metric='euclidean', optimal_ordering=True)
	fig, plot = g.run(**kwargs)
	return fig, plot