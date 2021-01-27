from util.plot_default import *

def plot_full_sample_collection(meta, ga_col_name, pp_col_name, plot_pp = True, term_order = [4,3,2,1], figsize = (10, 5)):
	'''
	Plotting utility to plot sample collection during and post-pregnancy with broken axis to denote the time split

	Inputs:
		meta - dataframe with metadata used for plotting
		ga_col_name - GA var column name in df (str)
		pp_col_name - PP collection var column name in df (str)
		plot_pp - Whether to plot PP collection [Not included in validation] (bool)
		term_order - Order to plot GA and PP collection on y-axis (str)
		figsize - figure size (tuple)
		

	Returns:
		fig handle for plot
	'''
	n_plots = 2 if plot_pp else 1
	fig, ax = plt.subplots(1, n_plots, sharey = True, figsize = figsize)

	ax_0 = ax[0] if n_plots > 1 else ax
	sns.swarmplot(x = ga_col_name, y = 'term', hue = 'case', dodge = True, orient = 'h', data = meta, order = term_order, 
				  size = 4, lw = 1.5, edgecolor = 'k', color = 'k', ax = ax_0)

	sns.violinplot(x = ga_col_name, y = 'term', hue = 'case', inner = 'stick', split = True, orient = 'h', cut = 0, data = meta, order = term_order,
				   ax = ax_0, palette = cntrl_pe_palette)

	if n_plots > 1:
		sns.swarmplot(x = pp_col_name, y = 'term', hue = 'case', dodge = True, orient = 'h', data = meta, order = term_order, 
					  size = 4, lw = 1.5, edgecolor = 'k', color = 'k', ax = ax[1])

		sns.violinplot(x = pp_col_name, y = 'term', hue = 'case',  inner = 'stick', split = True, orient = 'h', cut = 0, data = meta, order = term_order,
					   ax = ax[1], palette = cntrl_pe_palette)

	#Fix axes and legend

	plt.locator_params(axis="y", nbins=len(term_order))

	ax_0.set_yticklabels([term_labels[term_i] for term_i in term_order])
	ax_0.set_ylabel("")
	ax_0.set_xlabel("Gestational age (weeks)")

	handles, _ = ax_0.get_legend_handles_labels()
	ax_0.legend(handles, ['Normotensive', 'Preeclampsia'], ncol=2, frameon=0, bbox_to_anchor=(0.8, 1.1))

	if n_plots > 1:
		ax[1].set_ylabel('')
		ax[1].set_xlabel('Post-delivery (weeks)')
		ax[1].get_legend().remove()

		#Create broken axis based on https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/broken_axis.html
		#Zoom in
		ax_0.set_xlim([0, 35])
		ax[1].set_xlim([-0.2, 8]) #Go to 8 so tick spacing matches ish visually
		ax[1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))

		ax_0.tick_params(labelright = False)

		#Add diagonal tick marks
		d = .025  # how big to make the diagonal lines in axes coordinates
		kwargs = dict(transform=ax_0.transAxes, lw = 1, color='k', clip_on=False)
		ax_0.plot((1 - d, 1 + d), (- d, + d),  **kwargs)		# bottom-left diagonal
		ax_0.plot((1 - d, 1 + d),(1 - d, 1 + d), **kwargs)  # top-left diagonal

		kwargs.update(transform=ax[1].transAxes)  # switch to the bottom axes
		ax[1].plot((- d, + d), (1 - d, 1 + d), **kwargs)  # top-right diagonal
		ax[1].plot((- d, + d), (- d, + d), **kwargs)  # bottom-right diagonal

	#Bring subplots together
	fig.subplots_adjust(wspace = 0.05)
	fig.tight_layout()
	return fig