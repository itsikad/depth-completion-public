# Depth Completion Research Environment

A high-level environment for exploring, training and evaluating different architectures for depth-completion. 

Researchers from other tasks and fields may find it useful as a reference to start their own project.

It utilizes [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/) for easy scaling (single GPU to large clusters), [Hydra](https://hydra.cc/docs/intro/) for managing configs and [Neptune](https://neptune.ai/) for logging and managing experiments.

Implemented architectures:
1. [Guidenet](https://github.com/kakaxi314)
2. Supervised/Unsupervised [Sparse-to-Dense](https://github.com/fangchangma/self-supervised-depth-completion)
3. [ACMNet](https://github.com/sshan-zhao/ACMNet)

Datasets:
1. [KITTI Depth Completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)

Other architectures and deatasets can be added with only small modifications to existing code.

# Setup
1. Clone this repo
```bash
git clone https://github.com/itsikad/depth-completion-public.git
```

2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.tx
```

3. Edit [configs](./configs/):
    1. Set `data_root` datasets root path in [main config file](./configs/config.yaml)
    2. (Optional) Set `experiment_name`, `description` and `tags` fields for Neptune Logger
    3. Set `project_name` for Neptune Logger in [neptune logger config file](./configs/logger/neptune.yaml)
    4. Set number of gpus on your [debug machine](configs/machine/debug.yaml) or [server](configs/machine/server.yaml)

4. Currently, Neptune Logger doesn't integrate with Hydra so a simple corerction is required:
    1. in `.venv/lib/neptune/internal/api_cleints/hosted_api_clients/hosted_alpha_leaderboard_api_client.py' line 129 replace:
    ```python
    if not isinstance(params, Dict):
    ```
    with
    ```python
    if not isinstance(params, MutableMapping):
    ```

    2. in `.venv/lib/bravdo_core/schema.py' line 90 replace:
    ```python
    return isinstance(spec, (dict, Mapping))
    ```
    with
    ```python
    return isinstance(spec, (dict, typing.MutableMapping))
    ```

5. Set NEPTUNE_API_TOKEN environment variable, see [Neptune authentication](https://docs.neptune.ai/getting-started/installation)

# How To Use

## Train a model
Training an existing architecture is as simple as (example uses `guidenet`):

```bash
python src/run.py model=guidenet machine=server
```

where `guidenet` can be replaced with any other model located in `./configs/model/<model_name>.yaml`, for example `self_sup_sparse_to_dense`)

CAUTION: During the first run, KITTI Depth Completion dataset will be downloaded and processed. It might take several hours and roughly ~160gb of disk space.

## Add a new model / loss / dataset

Follow these steps to add a new model:

1. Your new model should base `models.base_model.BaseModel`, an abstract model.
2. It should return a dictionary, where the final depth prediction should be keyed by `pred`, other tensors used for debug, etc. can be also added to the dictionary.
3. Add your model to model builder in `model__init__.py`.
4. Model config should be placed in `./src/configs/model/<model_name>.yaml`
5. Train your model using:

```bash
python src/run.py model=<model_name> machine=server
```

Follow a similar process to add a new loss or dataset. For another dataset, don't forget to tell `Hydra` you're using a non-default dataset:

```bash
python src/run.py model=<model_name> machine=server dataset=<your_dataset>
```
