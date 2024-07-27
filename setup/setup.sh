source activate
conda env create -f setup/env.yml 
conda activate diffcsp_env
echo "current conda env: $(conda info --envs)"
conda install ipywidgets jupyterlab matplotlib pylint
conda install -c conda-forge matminer=0.7.3 nglview 
pip install setuptools==59.5.0 pymatgen==2023.8.10 torchmetrics==0.6.2
