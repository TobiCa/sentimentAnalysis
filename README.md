# sentimentAnalysis
Just a small attempt at creating a sentiment analysis model using pytorch


## Setup

This script has only been tested on python 3.8.2

Before running the script, please install the requirements from requirements.txt by running:

`pip install -r requirements.txt`

After that, please run:

`python -m spacy download en`

### Train


You can now run the following:

`python sentimentAnalysis.py train`

### Predict

After having trained the model, you can now run:

`python sentimentAnalysis.py predict 'string to predict sentiment of'`



