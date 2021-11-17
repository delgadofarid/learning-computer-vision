# learning-computer-vision
Demo on how to build a Dog/Cat recognizer from zero.

## Requirements

- Python >= 3.8

## How to install?

1. Install Jupyter Notebook:

```bash
$ pip install notebook
```

2. Run jupyter notebook from the root folder:

```bash
$ cd my-first-search-engine/notebook
$ jupyter notebook
```

3. Run each cell in the notebook `notebook/pipeline.ipynb`: The first cell(s) will install all the requirements in `notebook/requirements.txt`.


## How to run the identifier web app?

Before running this, you will need to populate the folder `app/model` with the both models, a Keras Sequence model and a Keras VGG16 model using the folder names `keras_seq_model` and `keras_vgg16_model` respectively, otherwise it will fail to run.

```bash
$ cd learning-computer-vision
$ env FLASK_APP=src/app python -m flask run
```