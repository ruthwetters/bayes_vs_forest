I declare that this is the sole work of Ruth Wetters, cited where appropriate.

In the folder 'ML_Submission':
	- Two folders : Data and Scripts
	- One .txt file : Readme.txt
	- One .pdf file : ml_2_.pdf

In the Data folder:
	- Two Folders : 
		Data : This contains "dfnamed1" (data file) and a "Data Description.txt" (data description file)
In the Scripts folder:
	- 5 .m files and 2 .mat files which are trained models

# Scripts
All scripts begin with the importing of the data file dfnamed1.csv. These currently include the original pathname, so in order to run the models, it is necessary for the user to insert the new pathname.

- featselect.m: a comparison of chi-squared and mrmr feature ranking methods, which produces one figure for each method

- naive_bayes.m: training file for naive bayes algorithm

- rf.m: training file for random forest algorithm

- nb.mat: trained NB model

- rf.mat: trained RF model

- RUNME1testnb: can be used by markers to run trained nb model on test data

- RUNME2testrf: can be used to run trained rf model on test data