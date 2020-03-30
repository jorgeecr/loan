# Credit risk model 

This script can be used to train, compare and evaluate a model and get predictions from it. 
The dataset consists of  credit reports of some clients from my old company (used with their permission). The dataset is perfectly balanced, but I've worked with very unbalanced datasets.
The objective is to classify good and bad clients, i.e. to predict the probability of default.

The first function get_features() creates features from the csv files. A lot of work was put into this part, and I believe feature engineering is the hardest part in any data science project. Three models are then trained and compared:
The first one is a logistic regression, which is still very used in the credit business, mainly because it's easy to interpret. The other two are different but related (between them) models: a Random forest classifier and a boosting trees ensemble, both have proven to give high accuracy. Although they lack in terms of interpretability, this can be overcome using other tools, for example the SHAP package.

The metric I chose to compare the models is the Area Under the Receiver Operating Characteristic Curve. This is a good metric for classification problems when they are balanced. I chose this metric because it's robust in the sense that it takes into account all the different thresholds of a binary classificator. 

I've chosen this script because it's related to past work I've actually done. In actual situations, I had more data, not just bureau reports, but I also had more restrictions. This piece shows all the aspects that one must consider in a machine learning application.

<br>

## Dependencies

```
sudo pip install xgboost sklearn pandas numpy
```

## Training
```
usage: model.py [-h] [-o OUTPUT] train input 

positional arguments:
  input                 Directory containing the training set

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Directory where the model is saved after the training.
 
```
The training dataset consists of a csv containing both the user list (users.csv) and their credit reports (credit_reports.csv) in a  predefined format. I've included both files in the data/training/ directory. 

You should use . to get to the current directory, e.g. 
```
python model.py train data/training/ -o .
```

## Predictions
```
usage: model.py [-h] [-o OUTPUT] predict input 

positional arguments:
  input                 Directory containing the sets for prediction

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Directory where the model is located
 
```
The input dataset should be a csv containing both the user list (users.csv) and the credit reports (credit.csv). The program will save a list of predictions in csv format in the same directory.

<br><br>
## Authors

* **Jorge Contreras** - jorgeecr@gmail.com
