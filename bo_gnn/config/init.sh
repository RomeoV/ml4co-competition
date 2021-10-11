# remove any previous environment
conda env remove -n ml4co

# create the environment from the dependency file
conda env create -n ml4co -f conda.yaml

conda install -c conda-forge -y mamba
mamba clean -a -y

mamba install pytorch cudatoolkit=11.1 -c pytorch -c nvidia -c conda-forge -y
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
pip install torch-geometric
mamba install -c conda-forge pytorch-lightning -y

conda activate ml4co

# additional installation commands go here

conda deactivate
