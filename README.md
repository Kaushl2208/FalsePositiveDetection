This repository is intented to work towards building a first class False positives detection method.


**Documentation about the project and weekly progress reports can be found on this link:**[Fossology GSoC](https://fossology.github.io/gsoc/)


## Using the Jupyter Notebook

1. Clone the repository using `git clone git@github.com:Kaushl2208/FalsePositiveDetection.git`

2. You can install Jupyter Notebook in your system or Use Jupyter Notebook Extension in VS Code.

## Input file type and Expected changes:
1. Input file should be `.csv` file characteristics: 
    a. "Copyright": That contains the copyright statements
    b. "Manual Tag" (optional): If you want to calculate the accuracy over manual tagging.
2. One flag should be provided `clutter_flag`, Which tells the script to remove the unwanted clutter from the TP copyright statements.

## Output File:
1. The output file will be `updatedChanges.csv` containing:
    a. *"Hit&Miss"*:  which tells us about the algorithm's output, ***t*** for a true copyright statement and ***f*** for a false copyright statement.
    b. *"edited_text"* : if the `clutter_flag` was true, The updated text without clutter will be seen here.


## Miscellaneous
1. You can also train the model over for your specific wordset/dataset/knowledge bag.
2. Provide the dataset in the spacy input format and run the `model_train.py` script. 
    a. You can also tune in the number of epochs and even train your own model with name in `modelName` variable.
    b. Normally it will train the pre used `en_core_web_sm` model over the new training set and may result into more accuracy for specific set.
 





