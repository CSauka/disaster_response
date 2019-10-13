# Disaster Response Pipeline
Project for Data Science class at Udacity


### Contents

The project provides ...
1. helper functions which clean and save text messages to be used in training a classifier (`process_data.py`)
2. a multi-output classification pipeline which transforms text messages and assigns these messages to categories (`train_classifier.py`)
3. a web application (`run.py` and html files) which ...
    - visualizes some statistics of the text data used for training
    - allows the user to input messages which then are classified using the trained model

<br>

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and stores the cleaned data in a SQLite database
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains a multi-class classifier and saves the trained model
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

<br>

### Inputs and outputs - overview
1. **ETL pipeline** (`process_data.py`):
    - **Inputs**: Expects (1) messages (`disaster_messages.csv`) and (2) the categories these messages belong to (`disaster_categories.csv`), each in a predefined structure
    - **Output**: Writes the cleaned, categorized message texts into a database (`DisasterResponse.db`)

2. **ML pipeline** (`train_classifier.py`):
    - **Input**: Expects a database containing cleaned, categorized message texts (`DisasterResponse.db`)
    - **Output**: Pickles the trained classifier (`classifier.pkl`)

3. **Web app** (`run.py`, `master.html`, `go.html`)
    - **Inputs**: Expects 
        - the message texts (`DisasterResponse.db`)
        - the classifier (`classifier.pkl`)
        - a message to be classified which is input by the user on the web site
    - **Outputs**: Web application showing (1) the structure of the messages used to train the classifier and (2) the categories the input message is assigned to by the classifier 

<br>

### Contemplations regarding the structure of the dataset
The dataset used to train the classifier is highly imbalanced. This has consequences for the interpretation of model evaluation metrics and may allow for improvements of the implemented model.

A. **Interpretation of classification scores**<br>
`train_classifier.py` prints out model testing scores for each message category. Due to the class imbalances classification accuracy in particular is likely to be an unreliable evaluation metric. Accuracy is prone to be driven largely by the imbalances in the dataset instead of reflecting the reliability of the model. The F1 score seems to be a more sensible measure. The F1 score could be substituted by a F-beta score. The latter can be specifically tuned into the relationship between the cost of overlooking a disaster and the cost of a false alarm. Alternatively, using adjusted evaluation metrics such as sklearn's balanced_accuracy_score could be considered.

B. **Potential improvements of the model**<br>
The fitting of the model could likely be improved by employing resampling techniques specifically designed to handle imbalanced data. Methods such as under-sampling or over-sampling could be considered in particular - see e.g. the [imbalanced-learn package](https://imbalanced-learn.org/en/stable/index.html).

### Acknowledgements
Many thanks to **Figure Eight** for sharing the disaster messages and their classifications used in this project!
