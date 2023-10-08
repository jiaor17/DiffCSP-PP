# DiffCSP

Implementation codes for submission 3458 (Crystal Structure Prediction by Joint Equivariant Diffusion on Lattices and Fractional Coordinates).

### Dependencies

```
python==3.8.13
torch==1.9.0
torch-geometric==1.7.2
pytorch_lightning==1.3.8
pymatgen==2022.9.21
```

### Training

For the CSP task

```
python diffcsp/run.py data=<dataset> expname=<expname>
```

For the Ab Initio Generation task

```
python diffcsp/run.py data=<dataset> model=diffusion_w_type expname=<expname>
```

The ``<dataset>`` tag can be selected from perov_5, mp_20, mpts_52 and carbon_24.
