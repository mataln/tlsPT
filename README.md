# tlsPT

This is the scratch implementation of PointMAE (arxiv.org/abs/2203.06604) that I used to generate the results for my presentation at AGU2024. This doesn't belong to a paper but I've made it public so I can share it with people who've asked.

You can see read some fast-bullets about my findings, or see the slides [here](https://docs.google.com/presentation/d/1nZ2_TzjnOq7FMeOqkhAYEYjJ7wfqbaLoSft4on0UXvE/edit?usp=sharing). I delivered most of this work from scratch within ~6-8 weeks of the conference, so the graphics aren't super polished, but the information is there.

Findings:
-MAE-style pretraining on patches of fixed radius improved performance in all cases.
-The improvement was similar or *greater* on regions that *weren't* in the pretraining data.
-Custom LR scheduler with gradual unfreezing improved performance quite a lot, likely due to not many labels.


![Alt text for the image](splash.PNG)

## Installation

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html) and then run the following commands to create the tlspt environment:

```bash
conda env create -f environment.yml

conda activate m3leo-env
```

Next, install the package:

```bash
pip install -e .
```

or if you want development dependencies as well:

```bash
pip install -e .[dev]
```

### Optional, but highly recommended

Install [pre-commit](https://pre-commit.com/) by running the following command to automatically run code formatting and linting before each commit:

```bash
pre-commit install
```

If using pre-commit, each time you commit, your code will be formatted, linted, checked for imports, merge conflicts, and more. If any of these checks fail, the commit will be aborted.

## Adding a new package

To add a new package to the environment, open pyproject.toml file and add the package name to "dependencies" list. Then, run the following command to install the new package:

```bash
pip install -e . # or .[dev]
```

```bash
CACHE_DIR=/path/to/cache/dir
```

## Running train.py
The training script is parameterised using hydra. You can see the existing configs under <configs/example-config>.

The training script can then be run using

```bash
python train.py --config-path /path/to/config
```
