from util.plot_default import *
import numpy as np

def plot_both_discovery_val_sample_collection(meta_dict):
	'''
	Fxn to visualize both discovery and val datasets
	Fairly specific to requirements for these datsets but can generalize in future

	Inputs:
		meta_dict: Dictionary with dataset name and kwargs to pass to plot_full_sample_collection
		n_plots: Total plot number
	'''
	init_fig = True
	
	n_rows = len(meta_dict.keys())
	curr_row = 0

	for ds_name, vals in meta_dict.items():

		if init_fig:
			fig, ax = plot_full_sample_collection(**vals, n_rows = n_rows)
			init_fig = False
		else:
			plot_full_sample_collection(**vals, to_plot_as_subplot = True, fig_to_use = fig, ax_handle_to_use = ax, ax_idx_to_use = curr_row)
		
		ax_0 = ax[curr_row, 0] if vals['plot_pp'] else ax[curr_row]

		ax_0.set_title(ds_name, loc = 'center')
		
		curr_row += 1

	plt.suptitle('Sampling time (weeks)', x = 0.6, y = -.03, va = 'bottom')

	return fig, ax

def plot_full_sample_collection(meta, ga_col_name = 'ga_at_collection', pp_col_name = 'weeks_post_del', plot_pp = True, term_order = [4,3,2,1], figsize = (10, 10), 
								n_rows = 2, 
								to_plot_as_subplot = False, fig_to_use = None, ax_handle_to_use = None, ax_idx_to_use = None):
	'''
	Plotting utility to plot sample collection during and post-pregnancy with broken axis to denote the time split

	Inputs:
		meta - dataframe with metadata used for plotting
		ga_col_name - GA var column name in df (str)
		pp_col_name - PP collection var column name in df (str)
		plot_pp - Whether to plot PP collection [Not included in validation] (bool)
		term_order - Order to plot GA and PP collection on y-axis (str)
		figsize - figure size (tuple)
		n_rows - number of rows
		to_plot_as_subplot - plot using passed ax handle
		fig_handle_to_use - fig handle
		ax_handle_to_use - ax handle
		ax_idx_to_use - ax idx to plot in

	Returns:
		fig handle for plot
	'''
	n_cols = 2 if plot_pp else 1

	if to_plot_as_subplot:
		fig = fig_to_use
		ax = ax_handle_to_use
	else:
		ratios = [3, 1] if n_cols == 2 else [3]
		fig, ax = plt.subplots(n_rows, n_cols, sharey = True, sharex = 'col', figsize = figsize, gridspec_kw={'width_ratios': ratios})
		ax_idx_to_use = 0

	ax_0 = ax[ax_idx_to_use, 0] if n_cols > 1 else ax[0]
	ax_1 = ax[ax_idx_to_use, 1] if n_cols > 1 else ax[1]

	sns.swarmplot(x = ga_col_name, y = 'term', hue = 'case', dodge = True, orient = 'h', data = meta, order = term_order, 
				  size = 4, lw = 1.5, edgecolor = 'k', color = 'k', ax = ax_0)

	sns.violinplot(x = ga_col_name, y = 'term', hue = 'case', inner = 'stick', split = True, orient = 'h', cut = 0, data = meta, order = term_order,
				   ax = ax_0, palette = cntrl_pe_palette)

	if plot_pp:
		sns.swarmplot(x = pp_col_name, y = 'term', hue = 'case', dodge = True, orient = 'h', data = meta, order = term_order, 
					  size = 4, lw = 1.5, edgecolor = 'k', color = 'k', ax = ax_1)

		sns.violinplot(x = pp_col_name, y = 'term', hue = 'case',  inner = 'stick', split = True, orient = 'h', cut = 0, data = meta, order = term_order,
					   ax = ax_1, palette = cntrl_pe_palette)

	#Fix axes and legend

	plt.locator_params(axis="y", nbins=len(term_order))

	ax_0.set_xlim(left = 0)
	ax_0.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))

	ax_0.set_yticklabels([term_labels[term_i] for term_i in term_order])
	ax_0.set_xlabel("")
	ax_0.set_ylabel("")

	if ax_idx_to_use == 1 or n_rows == 1:
		ax_0.set_xlabel("Gestational age")		

	if ax_idx_to_use == 0:
		handles, _ = ax_0.get_legend_handles_labels()
		ax_0.legend(handles, ['Normotensive', 'Preeclampsia'], ncol=2, frameon=0, bbox_to_anchor=(1, 1.15))

	if to_plot_as_subplot:
		ax_0.get_legend().remove()

	if plot_pp:
		ax_1.set_ylabel('')
		ax_1.set_xlabel('Post-delivery') if ax_idx_to_use == 1 or n_rows == 1 else ax_1.set_xlabel("")
		ax_1.get_legend().remove()

		#Create broken axis based on https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/broken_axis.html
		#Zoom in
		#ax_1.set_xlim([-0.2, 8]) #Go to 8 so tick spacing matches ish visually
		ax_1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

		ax_0.tick_params(labelright = False)

		#Add diagonal tick marks
		d = .025  # how big to make the diagonal lines in axes coordinates

		kwargs = dict(transform=ax_0.transAxes, lw = 1, color='k', clip_on=False)
		ax_0.plot((1 - d, 1 + d), (- d, + d),  **kwargs)		# bottom-left diagonal
		ax_0.plot((1 - d, 1 + d),(1 - d, 1 + d), **kwargs)  # top-left diagonal

		kwargs.update(transform=ax_1.transAxes)  # switch to the bottom axes
		ax_1.plot((- d, + d), (1 - d, 1 + d), **kwargs)  # top-right diagonal
		ax_1.plot((- d, + d), (- d, + d), **kwargs)  # bottom-right diagonal

	#Bring subplots together
	fig.subplots_adjust(wspace = 0.01)
	fig.tight_layout()
	return fig, ax