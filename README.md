# Early prediction of preeclampsia in pregnancy with cell-free RNA

# Overview
Repository to reproduce analyses from manuscript titled "Early prediction of preeclampsia in pregnancy with cell-free RNA"

# Getting started

1. Install miniconda if you do not have it already - https://docs.conda.io/en/latest/miniconda.html
2. Git clone this repo
3. Change into this directory
4. For analyses in python, create a virtual environment with required packages. 
	> conda env create -f envs/gen_comp.yaml 
5. For analyses in R, see "R_session_info.txt" for package requirements.

# Download data

* Counts table data can be found under GEO X 
	* There are two versions of counts table:
		1. Raw counts table includes all possible annotations based on gtf file
		2. Post QC counts table includes genes that passed filtering (see Methods, "Gene filtering")
* Raw data can be found under SRA Y
* Metadata can be found under SRA Y

# To run code

* To run any notebook starting with a number greater than 0, you will need counts table version b. The notebook will load any required utilities. 
