<div align="center">

# Meta-AF: Meta-Learning for Adaptive Filters
[Jonah Casebeer](https://jmcasebeer.github.io), [Nicholas J. Bryan](https://ccrma.stanford.edu/~njb/), and [Paris Smaragdis](https://paris.cs.illinois.edu/)

</div>

## Abstract
Adaptive filtering algorithms are pervasive throughout modern society and have had a significant impact on a wide variety of domains including audio processing, biomedical sensing, astropyhysics, and many more. Adaptive filters typically operate via specialized online, iterative optimization methods but can be laborious to develop and require domain expertise. In this work, we frame the development of adaptive filters as a deep meta-learning problem and present a framework for learning online, adaptive signal processing algorithms or update rules directly from data using self-supervision. We focus on audio applications and apply our approach to system identification, acoustic echo cancellation, blind equalization, multi-channel dereverberation, and beamforming. For each application, we compare against common baselines and/or state-of-the-art methods and show we can learn high-performing adaptive filters that operate in real-time and, in most cases, significantly out perform specially developed methods for each task using a single general-purpose configuration of our method. 

For more details, please see:
"[Meta-AF: Meta-Learning for Adaptive Filters](https://arxiv.org/abs/tbd)", [Jonah Casebeer](https://jmcasebeer.github.io), [Nicholas J. Bryan](https://ccrma.stanford.edu/~njb/), and [Paris Smaragdis](https://paris.cs.illinois.edu/), arXiv, 2022.

```BibTex
@article{casebeer2022meta,
  title={Meta-AF: Meta-Learning for Adaptive Filters},
  author={Casebeer, Jonah and Bryan, Nicholas J. and Smaragdis, Paris},    
  year={2022},
  url={https://arxiv.org/abs/tbd}
}
```
If you use ideas or code from this work, pleace cite our paper.


<div align="center">
</br>


[**Demos**](#demos)
| [**Code**](#code)
| [**Meta-AF Zoo**](#meta-af-zoo)
| [**License**](#license)
| [**Related Works**](#related-works)


</br>
</div>

 



## Demos

For audio demonstrations of the work and `metaaf` package in action, please check out our [demo website](https://jmcasebeer.github.io/projects/metaaf/). You'll be able to find demos for the five core adaptive filtering problems.

## Code

We open source all code for the work via our `metaaf` python pip package. Our `metaaf` package has functionality which enables meta-learning optimizers for near-arbitrary adaptive filters for any differentiable objective. `metaaf` automatically manages online overlap-save and overlap-add for single/multi channel and single/multi frame filters. We also include generic implementations of LMS, RMSProp, NLMS, and RLS for benchmarking purposes. Finally, `metaaf` includes implementation of generic GRU based optimizers, which are compatible with any filter defined in the `metaaf` format. Below, you can find example usage, usage for several common adaptive filter tasks (in the adaptive filter zoo), and installation instructions.

The `metaaf` package is relatively small, being limited to a dozen files which enable much more functionality than we demo here. The core meta-learning code is in `core.py`, the buffered and online filter implementations are in `filter.py`, and the RNN based optimizers are in `optimizer_gru.py` and `optimizer_fgru.py`. The remaining files hold utilities and generic implementations of baseline optimizers. `meta.py` contains a class for managing training. 

### Installation

To install the `metaaf` python package, you will need a working JAX install. You can set one up by following the official directions [here](https://github.com/google/jax#installation). Below is an example of the commands we use to setup a new conda environment called `metaenv` in which we install `metaaf` and any dependencies.

```{bash}
# Install all the cuda and cudnn prerequisites
conda create -yn metaenv python=3.7 &&
conda install -yn metaenv cudatoolkit=11.1.1 -c pytorch -c conda-forge &&
conda install -yn metaenv cudatoolkit-dev=11.1.1 -c pytorch -c conda-forge &&
conda install -yn metaenv cudnn=8.2 -c nvidia -c pytorch -c anaconda -c conda-forge &&
conda install -yn metaenv pytorch cpuonly -c pytorch -y
conda activate metaenv

# Actually install jax
# You may need to change the cuda/cudnn version numbers depending on your machine
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html  

# Install Haiku
pip install git+https://github.com/deepmind/dm-haiku
```

Finally, with the prerequisites done, you can install `metaaf` by running `pip install git+https://github.com/adobe-research/metaaf`. You could also clone the repo, move into this directory and run `pip install -e ./`. This `pip install` adds the remaining dependencies.

### Example Usage

The `metaaf` package provides several important modules to facilitate training. The first is the `MetaAFTrainer`, a class which manages training. To use the `MetaAFTrainer`, we need to define a filter architecture, and a dataset. `metaaf` adopts several conventions to simplify training and automate procedures like buffering. In [this notebook](examples/sysid_demo.ipynb), we walk through this process and demonstrate on a toy system-identification task. In this section, we explain that toy-task and the automatic argparse utilities. To see a full-scale example, proceed to the next section, where we describe the Meta-AF Zoo.

First, you need to make a datatset using a regular PyTorch dataset. The dataset must return a dictionary with two keys: "signals" and "metadata". The "signals" are automatically indexed and sliced and should be of size samples by channels.

```{python}
class SystemIDDataset(Dataset):
    def __init__(self, N=4096, sys_order=32):
        self.N = N
        self.sys_order = sys_order

    def __len__(self):
        return 256

    def __getitem__(self, idx):
        # the system
        w = np.random.normal(size=self.sys_order) / self.sys_order

        # the input
        u = np.random.normal(size=self.N)

        # the output
        d = np.convolve(w, u)[: self.N]

        return {
            "signals": {
                "u": u[:, None], # time X channels
                "d": d[:, None], # time X channels
            },  
            "metadata": {},
        }
train_loader = NumpyLoader(SystemIDDataset(), batch_size=32)
val_loader = NumpyLoader(SystemIDDataset(), batch_size=32)
test_loader = NumpyLoader(SystemIDDataset(), batch_size=32)

```

Then, you define your filter. We're going to inherit from the `metaaf` OLS module. When inheriting, you can return either the current result, which will be automatically buffered, or a dictionary. When returning a dictionary it must have a key "out" which will be buffered. All other keys are stacked and returned.

```{python}
from metaaf.filter import OverlapSave
# the filter inherits from the overlap save modules
class SystemID(OverlapSave, hk.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # select the analysize window
        self.analysis_window = jnp.ones(self.window_size)

    # Since we use the OLS base class, x and y are stft domain inputs.
    # The filter msut take the same inputs provided in its _fwd function.
    def __ols_call__(self, u, d, metadata):
        # collect a buffer sized anti-aliased filter
        w = self.get_filter(name="w")

        # this is n_frames x n_freq x channels or 1 x F x 1 here
        return (w * u)[0]
    
    @staticmethod
    def add_args(parent_parser):
        return super(SystemID, SystemID).add_args(parent_parser)

    @staticmethod
    def grab_args(kwargs):
        return super(SystemID, SystemID).grab_args(kwargs)
```

Haiku converts objects to functions. We need to provide a wrapper to do this. The wrapper function MUST take as input the same named values from your dataset.

```{python}
def _SystemID_fwd(u, d, metadata=None, init_data=None, **kwargs):
    f = SystemID(**kwargs)
    return f(u=u, d=d)
```

Then, we define an adaptive filter loss. Here, just the MSE. An adaptive filter loss must be written in this form, so that `metaaf` can automatically take its gradient and pass it around.

```{python}
def filter_loss(out, data_samples, metadata):
    e =  out - data_samples["d"]
    return jnp.vdot(e, e) / (e.size)
```

We can construct the meta-loss in a similar fashion.

```{python}
def meta_loss(losses, outputs, data_samples, metadata, outer_learnable):
    EPS = 1e-9
    return jnp.log(jnp.mean(jnp.abs(outputs - data_samples["d"]) ** 2) + EPS)
```

With everything defined, we can setup the Meta-Trainer and start training.

```{python}
# Collect arguments
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="")
parser = ElementWiseGRU.add_args(parser)
parser = SystemID.add_args(parser)
parser = MetaAFTrainer.add_args(parser)
kwargs = vars(parser.parse_args())

# Setup trainer
system = MetaAFTrainer(
    _filter_fwd=_SystemID_fwd,
    filter_kwargs=SystemID.grab_args(kwargs),
    filter_loss=filter_loss,
    meta_train_loss=meta_loss,
    meta_val_loss=meta_loss,
    optimizer_kwargs=ElementWiseGRU.grab_args(kwargs),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)
# Train
key = jax.random.PRNGKey(0)
outer_learned, losses = system.train(
    **MetaAFTrainer.grab_args(kwargs),
    key=key,
)
```

That is it! For more advanced options check out the zoo, where we demonstrate call backs, customized filters, and more.

## Meta-AF Zoo

The [Meta-AF Zoo](zoo/README.md) contains implementations for system identification, acoustic echo cancellation, equalization, weighted predection error dereverberation, and a generalized sidelobe canceller beamformer all in the `metaaf` framework. You can find intructions for how to run, evaluate, and setup those models [here](zoo/README.md). For trained weights, please see the tagged release tar ball.



## License

All core utility code within the `metaaf` folder is licensed via the [University of Illinois Open Source License](metaaf/LICENSE). All code within the `zoo` folder and model weights are licensed via the [Adobe Research License](zoo/LICENSE). Copyright (c) Adobe Systems Incorporated. All rights reserved.



## Related Works

Please also see an early version of this work:

"[Auto-DSP: Learning to Optimize Acoustic Echo Cancellers](https://arxiv.org/abs/2110.04284)", [Jonah Casebeer](https://jmcasebeer.github.io), [Nicholas J. Bryan](https://ccrma.stanford.edu/~njb/), and [Paris Smaragdis](https://paris.cs.illinois.edu/), arXiv, 2022.

```BibTex
@inproceedings{casebeer2021auto,
  title={Auto-DSP: Learning to Optimize Acoustic Echo Cancellers},
  author={Casebeer, Jonah and Bryan, Nicholas J. and Smaragdis, Paris},
  booktitle={2021 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  pages={291--295},
  year={2021},
  organization={IEEE}
}
```
