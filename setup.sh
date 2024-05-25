# Install Python dependencies
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install -c pyg pytorch-sparse -y
conda install -c pyg pytorch-scatter -y
conda install -c pyg pytorch-cluster -y
conda install -c pyg pyg -y
pip install ogb
conda install -c dglteam/label/cu113 dgl -y
pip install yacs
pip install transformers
pip install --upgrade accelerate

# For typing
pip install pandas-stubs

# For downloading the datasets
pip install gdown

# # For spectral embedding
# pip install julia
# pip install h5py
# pip install networkx
# pip install python-louvain