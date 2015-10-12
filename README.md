# Introduction

This model has been trained from the MP3 Training Dataset. CMU's pocketsphinx has been used to convert the speech to text. On the text various classifiers have been applied. 

After the benchmarking, Passive Aggressive Classifier is considered to be best suited for training model. 

### Requirements

Following python modules are required for running the sample.py

1) numpy

2) pocketsphinx

3) sphinxbase

4) sklearn

### Running the script

Once you've cloned this repo and all the above requirements are met.

1) Place all the mp3 files in the main folder containing the sample.py file.

2) Run the sample.py file (python sample.py) or run the shell script run.sh (./run.sh).

### Trainig dataset

From mp3 the text converted data for training the model is in the file TrainingSet.txt

### Output

Two output file's are being generated:

1) results.txt > predicted sentiment separated by line

2) filename_pred.txt > a map with file name and it's predicted sentiment like one.mp3->unhappy
