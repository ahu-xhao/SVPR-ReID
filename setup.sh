# create environment
conda create -n svpr_reid python=3.10 -y
conda activate svpr_reid

# install pytorch and torchvision
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

