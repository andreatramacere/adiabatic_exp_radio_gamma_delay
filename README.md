

This repository contains the notebooks  to reproduce the analysis published in the paper....
The notebooks and the code in the folder: `jetset_sim_and_analysis` reproduce the analysis presented 
in Section 3,4,5, and 7.
The notebooks and the code in the folder: `obs_data_analysis` reproduce  the analysis presented 
in Section 6.


## Instructions to run the notebook in the `jetset_sim_and_analysis`
The version of jetset is 1.2.0rc8:
https://github.com/andreatramacere/jetset/releases/tag/1.2.0rc8

Since this is a prerelease, it is not published yet on conda and pip,
to install this version, yuo can use one of the methods described below:

- use the install script in : https://github.com/andreatramacere/jetset-installer

OR

- install from source or binaries:
    1) from binaries: https://github.com/andreatramacere/jetset/wiki/install-release-from-binaries

    2) from source: https://github.com/andreatramacere/jetset/wiki/install-release-from-source

### Instructions to run the notebooks


The directory `expansion_tools` hosts some helper modules (no need to install), you just need to have this directory in the same directory where you run the notebooks

All the  products of the simulations are already distributed, if  you want to  rerun the simulations (hosted in the directories with the `_sim` string ), yuo can set `run=True` in cells of the notebook where it is defined. You will find a corresponding description in each cell of  the simulation notebooks. If you want to rerun the simulations, please notice that products generated in 1 (`Flare_sim.ipynb`) are necessary to run 2  (`Expansion_sim_exp_vs_no_exp.ipynb`)  and 3(`Expansion_sim_exp_vs_no_exp.ipynb`).

List of the notebooks, and description:

1) `Flare_sim.ipynb` This notebook is in charge to run the simulation for the flaring event, Section 3.2 and Figure 2

2) `Expansion_sim_exp_vs_no_exp.ipynb` This notebook is in charge to run the simulations for the comparison of expansion vs non-expansion, Section 4 and Figure 3 and Figure 4

3) `Expansion_sim.ipynb` This notebook is in charge to run the simulations for
the long-term adiabatic expansions used in Section 5

4) `Rad_Adb_cooling.ipynb` This notebook is in charge to reproduce the anlysis regarding radiative/adiabatic cooling competition, in Section 5, Figure 5 and Figure 6

5) `Expansion_analysis_beta_exp_trends.ipynb` This notebook is in charge to reproduce the analysis for the long-term adiabatic expansions, beta exp trend, used in Section 5 and 5.1 of the paper Fig.7 and Fig. 8

6) `Expansion_analysis_beta_exp_0.1_nu_trends.ipynb` This notebook is in charge to reproduce the analysis for the long-term adiabatic expansions, nu trend, used in Section 5.2 of the paper Fig.9 

7) `Estimate_par_Mrk501_MRk421.ipynb`  This notebook is in charge to reproduce the analysis in the Discussion section of the paper, Fig. 13

7) `BlobExpGeom.ipynb`  This notebook is in charge to reproduce the analysis in the Discussion section of the paper, Fig. 14



## Instructions to run the notebook in the `obs_data_analysis`
