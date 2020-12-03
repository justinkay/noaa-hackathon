# noaa-hackathon

## Setup
```
conda create -n noaa python=3.7
conda activate noaa
pip install -r requirements.txt \ 
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
```

To add the conda env as a Jupyter kernel:
```
conda activate noaa
python -m ipykernel install --user --name noaa
```

Notes: 
- assumes CUDA 10.1, and will install pytorch 1.6
