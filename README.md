# HHNK-dijkprofiel
Automatic Dijkprofile annotation. `dijkprofile_annotation.ipynb` can be used to train the prediction network and generate the annotations.
the `generateAnnotations.py` standalone script can be used as well if the model is present.

A docker image that can be used as a standalone tool is also available at https://hub.docker.com/r/jonathangerbscheid/profile-annotator.
Just place the surfaceline.csv file in a folder, inject the folder as `/data` into the container and run it as such:
`docker run -v /path/to/data/folder:/data jonathangerbscheid/profile-annotator`
The resulting characteristic points file will be saved to the data folder as `annotations.csv`
