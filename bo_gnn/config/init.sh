# remove any previous environment
conda env remove -n ml4co

# create the environment from the dependency file
conda env create -n ml4co -f conda.yaml

conda activate ml4co

# additional installation commands go here
pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.1+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.9.1+cpu.html
pip install torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html

conda deactivate
