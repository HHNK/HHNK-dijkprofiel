# HHNK-dijkprofiel
This repository contains code for annotating dijk profiles, specifically the profiles generated for use with the [qDAMEEdit tool](http://www.twisq.nl/qdamedit/manual.pdf) The approach taken to determine the characteristic points is based on applying Unet in 1D.

# Code
`dijkprofile_annotation_full.ipynb` can be used to train the segmentation network and generate the annotations. <br>
`dijkprofile_annotation_simple.ipynb` can be used to generate the annotations from a notebook if the weights and scalers are present,
the `generateAnnotations.py` standalone script can be used for this as well if the model weights and scalers are present. <br>

## Docker
### Building the image from the repository
To build the docker image and run the tool, first clone the repository: <br>
`git clone https://github.com/HHNK/HHNK-dijkprofiel.git` <br>
Then (if you have docker installed) build the image with: <br>
`docker build -t dijkprofile_annotator .` <br>
The image can then be ran as a standalone tool as such: <br>
`docker run -v /path/to/local/data/folder:/data dijkprofile_annotator` <br>
Where the local data folder contains a surfaceline file named `surfacelines.csv`. The resulting characteristic points file will be saved to the data folder as `charpoints_scriptgenerated.csv` 

### Using image from dockerhub
A docker image that can be used as a standalone tool is also available at https://hub.docker.com/r/jonathangerbscheid/profile-annotator.
Just place the surfaceline.csv file in a folder, inject the folder as `/data` into the container and run it as such:
`docker run -v /path/to/local/data/folder:/data jonathangerbscheid/profile-annotator`
The resulting characteristic points file will be saved to the data folder as `charpoints_scriptgenerated.csv` 
