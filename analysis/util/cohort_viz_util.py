from util.plot_default import *

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