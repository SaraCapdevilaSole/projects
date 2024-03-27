# NLP Coursework Repository

This is the repository for the Natural Language Processing (NLP) coursework. The codebase is organized into a structured hierarchy, as described below.
## Repository Structure

### 1. `src` Folder

The main files to be called are contained within this folder are:

#### a. `back_translate.py`
- Responsible for creating the back-translated augmented dataset.
- Utilizes methods from `utils/data_augmentation.py`.

#### b. `baselines.py`
- Executes baseline models.
- Data cleaning methods are implemented in `utils/data_cleaning.py`.

#### c. `main_analysis.py`
- Runs data analysis, with relevant scripts located in the `analysis` folder.

#### d. `main.py`
- Executes main models, including RoBERTa and BERT.
- Respective trainers, predictors, and augmentation pipelines are stored in the `utils` folder.

#### e. `ssmba_main.py`
- Creates the SSMBa data augmented dataset (with and without back translation).
- Requires the output of the SSMBa model, available in [1].

### 2. `utils` Folder

Contains utility scripts used across different components:

- `config.py`: Configurations file, containing e.g. paths to files and GPU initialisation.
- `data_augmentation.py`: Methods for data augmentation.
- `data_cleaning.py`: Methods for cleaning and preprocessing data.
- `predictor.py`: Methods for running evaluation and predicting the class labels for both dev and test,
- `data_loader.py`: Methods for loading the data and making it in the correct format.
- `train.py`: Methods for training the models, including callbacks and adjusting model configuration parameters.

### 3. `analysis` Folder

Houses scripts for data analysis. These include:
- `data_analyser.py`: Methods for calling and displaying data insights.
- `utils.py`: Methods for computing and plotting, which are called by `data_analyser.py`.

### 3. `models` Folder

Files for the different models including RoBERTa and BERT. There is also a `model_configurations.py` that houses the specific parameters used for each model.

### How to Run `main.py`

To execute the `main.py` script, follow the instructions below. Customize the parameters according to your requirements.

```bash
python main.py --model_name "model_name" \
               --mode "model_mode" \
               --save_as "path_to_your_directory"
```

- `--model_name`: Specify the model to run (e.g., ROBERTA, BERT, RANDOM).
- `--mode`: Choose the mode, whether to train or test the selected model.
- `--save_as`: Provide the path to the directory where the results will be saved.

Example:
```bash
python main.py --model_name "ROBERTA" --mode "train" --save_as "./output_directory"
```

## Notes

To run `ssmba_main.py`, ensure you have the output of the SSMBA model available. The code was taken from [1].

## Requirements

- `requirements.txt`: Lists necessary packages to run the code.

Install the required packages listed in the file using:
```bash
pip install -r requirements.txt
```

```
Nathan Ng, Kyunghyun Cho, and Marzyeh Ghassemi. 2020. Ssmba: Self-supervised manifold based data augmentation for improving out-of-domain robustness [1].
```
