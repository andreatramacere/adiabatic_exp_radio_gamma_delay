

This repository contains the notebooks reproduce the analysis published in the paper "Radio-γ-ray response in blazars as signature of adiabatic blob expansion"


# Instructions to run the notebooks in the `jetset_sim_and_analysis`

The notebooks and the code in the folder: `jetset_sim_and_analysis` reproduce the analysis presented in Section 2,3,4,5,7, and Appendix C.

For questions or support, please contact andrea.tramacere[at]gmail.com

The version of jetset to use is 1.2.0rc11:
https://github.com/andreatramacere/jetset/releases/tag/1.2.0rc11

Since this is a pre-release, it is not published yet on conda and pip,
to install this version, you can use one of the methods described below:

- use the installation script in : https://github.com/andreatramacere/jetset-installer

OR

- install from source or binaries:
    1) from binaries: https://github.com/andreatramacere/jetset/wiki/install-release-from-binaries

    2) from source: https://github.com/andreatramacere/jetset/wiki/install-release-from-source

For some notebooks you will need to install the uncertainties  package: http://pythonhosted.org/uncertainties/

## Instructions to run the notebooks 


The directory `expansion_tools` hosts some helper modules (no need to install), you just need to have this directory in the same directory where you run the notebooks

All the notebooks with simulations, already display the results of the simulations, but you have to rerun these notebooks, to be able to perform the analysis in the other notebooks.
To rerun the simulations (hosted in the notebooks with the `*_sim*` string, i.e. the one in items 1,2,3,4, and 5), you have to set `run=True` in cells of the notebook where this variable is defined. You will find a corresponding description in each cell of the simulation notebooks. Please, notice that products generated in 2 (`Flare_sim.ipynb`) are necessary to run 3  (`Expansion_sim_exp_vs_no_exp.ipynb`), 4 (`Expansion_sim.ipynb`) and 5 (`Expansion_sim_no_radiative_cooling.ipynb`).
Products generated in 6,7,8, and 9, are necessary to run the analysis in 10,11,12,13, and 14.

It is important to run the notebooks following the sequence as reported below.



List of the notebooks, and description:

1) `time_scales.ipynb` Notebook to reproduce Figure 2 in Section 2 

2) `Flare_sim.ipynb` Notebook to run the simulation for the flaring event, Section 3.2 and Figure 3

3) `Expansion_sim_exp_vs_no_exp.ipynb` Notebook to run the simulations for the comparison of expansion vs non-expansion case, Section 4, Figure 4 and Figure 5

4) `Expansion_sim.ipynb` Notebook to run the simulations with radiative+adiabatic cooling, necessary for the analysis in Section 5, Section 5.1, Section 5.2, and Section 5.3

5) `Expansion_sim_no_radiative_cooling.ipynb` Notebook to run the simulations with only adiabatic cooling, necessary for the analysis in Section 5.1

6) `Convolution_analysis_beta_exp_0.1_nu_trends_no_radiative_cooling.ipynb` Notebook to run the convolution analysis for the long-term simulations without radiative cooling and beta_exp=0.1, necessary for the analysis in Section 5.1

7) `Convolution_analysis_beta_exp_0.1_nu_trends.ipynb` Notebook to run the convolution analysis for the long-term simulations with both radiative and adiabatic cooling and beta_exp=0.1, necessary for the analysis in Section 5.1
and Section 5.3

8) `Convolution_analysis_beta_exp_trends_no_radiative_cooling.ipynb` Notebook to run the convolution analysis for the long-term simulations without radiative cooling and beta_exp ranging in [0.001,0.3], necessary for the analysis in Section 5.1

9) `Convolution_analysis_beta_exp_trends.ipynb` Notebook to run the convolution analysis for the long-term simulations with both radiative and adiabatic cooling and beta_exp ranging in [0.001,0.3], necessary for the analysis in Section 5.1 and 
 5.2, and to produce Figure 6


10) `Phenomenology_trends_validation.ipynb` Notebook to run the validation of the phenomenological trends Section 5.1 of the paper Figure 7 

11) `Rad_Adb_cooling.ipynb` Notebook to run Radiative/Adiabatic cooling ratio analysis, Section 5.1 of the paper, Figure 8 and Figure 9

12) `Expansion_analysis_A_trend_cooling.ipynb` Notebook to run the response amplitude (A) trends analysis, Section 5.1 Figure 10 

13) `Expansion_analysis_beta_exp_trends.ipynb` Notebook to run the analysis for the long-term adiabatic expansions, beta_exp trend, used in Section 5.2 of the paper Figure 11

14) `Expansion_analysis_beta_exp_0.1_nu_trends.ipynb` Notebook to run the analysis for the long-term adiabatic expansions, nu trend, used in Section 5.3 of the paper Figure 12

15) `Estimate_par_Mrk501_MRk421_3C273_MCMC.ipynb` Notebook to run the analysis MCMC analysis in section of the 7 paper, Figures: 16,17,18, and 19, and the validation in appendix C, Figures: C.1 and C.2 **Since the random seed has not been fixed, different runs can show slight different posterior distributions**

16) `BlobExpGeom.ipynb` Notebook to run the analysis for jet profile in the Section 7 of the paper, Fig. 20

# Instructions to run the notebooks in the `radio_gev_data_and_analysis`

The notebooks and the code in the folder: `radio_gev_data_and_analysis` reproduce  the analysis presented in Section 6.  For question regarding the following notebooks please contact vitalii.sliusar [at] unige.ch

The directory `radio_gev_data_and_analysis` hosts notebooks to analyse data (available in the `radio_gev_data_and_analysis/data` directory). If you want to rerun the analysis to obtain all the values from the manuscript, just "Run All" cells of the provided notebooks. Generated plots are put into `radio_gev_data_and_analysis/images` folder

List of the notebooks, and description:

1) `processing_mrk421.ipynb` This notebook is used to analyze Mrk 421 data. Figure 13 and data of Table 5 of Section 6.2 are generated using this notebook.
2) `processing_mrk501.ipynb` This notebook is used to analyze Mrk 501 data. Figure 14 and data of Table 6 of Section 6.3 are generated using this notebook.
3) `processing_3c273.ipynb` This notebook is used to analyze 3C 273 data. Figure 15 and data of Tables 7 and 8 of Section 6.4 are generated using this notebook.
