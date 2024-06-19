# DiffCSP++



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

The ``<dataset>`` tag can be selected from perov_5, mp_20, mpts_52 and carbon_24. Pre-trained checkpoints are provided [here](https://drive.google.com/drive/folders/1FQ_b6CE09KtyGaU_r6uO8_I5JhrQmUFB?usp=sharing).

### Evaluation

#### Crystal Structure Prediction 

```
python scripts/evaluate.py --model_path <model_path> --dataset <dataset>
python scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv 
```

#### Ab initio generation

```
python scripts/generation.py --model_path <model_path> --dataset <dataset>
python scripts/compute_metrics.py --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv
```

### Sample from pre-defined symmetries

One can sample structures given the symmetry information by the following codes:

```
python scripts/sample.py --model_path <model_path> --save_path <save_path> --spacegroup <spacegroup> --wyckoff_letters <wyckoff_letters> --atom_types <atom_types>
```

One example can be

```
python scripts/sample.py --model_path mp_csp --save_path MnLiO --spacegroup 58 --wyckoff_letters 2a,2d,4g --atom_types Mn,Li,O
# Or simplify WPs as
python scripts/sample.py --model_path mp_csp --save_path MnLiO --spacegroup 58 --wyckoff_letters adg --atom_types Mn,Li,O
```

If multiple structures are required to be generated, one can first collect the inputs into a json file like `example/example.json`:

```
[
    {
        "spacegroup_number": 58,
        "wyckoff_letters": ["2a","2d","4g"],
        "atom_types": ["Mn","Li","O"]
    },
    {
        "spacegroup_number": 194,
        "wyckoff_letters": "abff",
        "atom_types": ["Tm","Tm","Ni","As"]
    }
]
```

And parallelly generate the structures via

```
python scripts/sample.py --model_path <model_path> --save_path <save_path> --json_file <json_file>
```

Notably, the `atom_types` parameter is not required if a model for ab initio generation is used.