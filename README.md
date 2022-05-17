# SurfaceXplorerPython
Python GUI for transient absorption data correction and fitting

## updated 05/17/2022

Now GUI window can be resized

## updated 05/11/2022

Added/changed features:
1. Model filtering based on number of species user selected.
2. Multivariate Curve Resolution - Alternating Least Squares (MCR-ALS) fitting of data (using pyMCR package from NIST).
3. User choice of fitting population curves extracted from MCR-ALS analysis to the selected model or global analsysis fitting (fit entire deltaA matrix to model).
4. Different view types: 
  a. 'Points of interest' (kinetic curves and spectra at selected points on surface)
  b. 'Global fitting analysis' (Normalized populations and species' spectra extracted from model)
  c. 'MCR-ALS analysis' (Populations and species' spectra extracted from MCR-ALS analsysis with overlaid model populations)
5. Color bar for deltaA residual matrix
6. SVD filter now shows residual matrix instead of singular values.

## updated 03/04/2022

Now fits are done globally using the entire deltaA matrix instead of single-wavelength fitting as before.
This was found to produce much better fits than single wavelength fitting (however, this takes more time to fit).
Therefore, the fitting can be done even without selecting wavelengths by the user, and the selected wavelengths are
only for displaying the fits on the selected kinetic traces.

## updated 02/25/2022

Added features:
fitting to decay rate distributions ( A(dist)->Gnd, and A(dist)->B(dist)->Gnd )

## updated 01/23/2022

Added features:
1. SVD filtering
2. Time-zero fitting

## updated 10/08/2021

To be able to run "SurfaceXplorerPython" you need to have Python installed on your computer.
This version of the program was tested with the Anaconda Python Distribution 2021.05 and Python 3.8.
For other computer systems, please go to Anaconda website at
www.anaconda.com and download a version suitable for your computer.

If you want to install Python on your own, make sure you also have the following libraries:
1. Numpy
2. Scipy
3. Matplotlib

After installing Python open a command prompt and run the following command without quotes (in the folder where main.py is located):
"python main.py"
Alternatively, you can start a python IDE and run main.py from there... Whatever is more convenient.

For adding/changing models in the model library, open the file "model_library.py" in any text
editor and follow the instructions in the file header.

Happy fitting!!!

--Itai
