# code structure?
## core: 
	including implementations of shell deformation energy cost, gradient, and hessian derivations;
	including implementation of elastic average and geodesic average;
	including geodesic interpolation and extrapolation;
	including implementation of objective function for model fitting and mesh editing; 
	
## data:
	c3d_data:
		includes *.c3d type motion capture data selected from CMU MoCap repo and MPI MoSh website
	faust:
		includes 10 sample meshes taken from MPI FAUST dataset
	precomputed:
		includes some precomputed model files or data files for the usage of running some included demos

## util:
	c3d_matlab:
		includes scripts to load and read c3d files in matlab
	others:
		includes supportive codes for mesh loading and saving, etc

# any demo code to run?
Yes. We provide several demo scripts which can be used readily to reproduce the results as shown in our paper. With just a few modifications, you can work on your own data. 

- filename: 
**demo_GCA.m**
	this code implements the core part of our paper, it produces the average shape and a set of principal variations shapes, given a couple of meshes; the current version only supports working on low resolution meshes; stand-alone deformation transfer codes may be provided in the future;
  
requirements:
	a set of decimated meshes with the same topology
	
- filename: 
**demo_Recon.m**
	this code implements shape reconstruction using the trained PGA model
  
requirements:
	1. pre-computed PGA model
	2. test data with the same topology as training data
	
- filename: 
**demo_fitting_to_markers_.m**
	this code implements fitting the (human body) model to motion capture data (c3d file)
  
requirements:
	1. pre-computed PGA model
	2. motion capture data in c3d format
	
- filename: 
**demo_mesh_edit.m**
	this code implements soft-constraint mesh editing (hard-constraint may be released in the future)
  
requirements:
	1. pre-computed PGA model
	2. model correspondence indices to handles, and new handle positions
	3. (optional) training data of the model to be used as appropriate initialization
	
