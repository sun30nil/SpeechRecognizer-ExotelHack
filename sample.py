import sys
import os
from glob import glob
from os import environ, path
import numpy as np
import json

_outputResultTo = "results.txt"
_outputResultTo2 = "filename_pred.txt"


def decodeSpeech(hmmd, lmdir, dictp, wavfile):
    """
    Decoding a speech file
    """

    try:
        import sphinxbase
        import pocketsphinx as ps

    except:
        import pocketsphinx as ps
        print """Pocket sphinx and sphixbase is not installed
        in your system. Please install it with package manager.
        """
    speechRec = ps.Decoder(hmm=hmmd, lm=lmdir, dict=dictp)
    wavFile = file(wavfile, 'rb')
    speechRec.decode_raw(wavFile)
    result = speechRec.get_hyp()
    print result[0]
    return result[0]


def convertMp3ToWav(mp3File, destinationFileName):
    from pydub import AudioSegment
    sound = AudioSegment.from_mp3(mp3File)
    sound.export(destinationFileName, format="wav")


def read_mp3_files_from_current_dir():
    return glob(path.join(".", "*.mp3"))


def process_mp3_files():
    files = read_mp3_files_from_current_dir()
    results = ""
    results2 = ""
    for mp3File in files:
        outFileName = os.path.basename(mp3File).split(".")[0] + ".wav"
        convertMp3ToWav(mp3File, outFileName)
        MODELDIR = "pocketsphinx-python/pocketsphinx/model"
        hmdir = path.join(MODELDIR, 'en-us/en-us')
        lmd = path.join(MODELDIR, 'en-us/en-us.lm.dmp')
        dictd = path.join(MODELDIR, 'en-us/cmudict-en-us.dict')
        classText = decodeSpeech(hmdir, lmd, dictd, outFileName)
        # print classText
        predicted_class = compute_emotion(classText)
        print mp3File, "=>", predicted_class
        results += predicted_class + "\n"
        results2 += mp3File + "->" + predicted_class + "\n"
        try:
            os.remove(outFileName)
        except OSError:
            pass
    try:
        f1 = open(_outputResultTo, 'w')
        f1.write(results)
        f2 = open(_outputResultTo2, 'w')
        f2.write(results2)
    except IOError:
        pass


def trainOnModel(x_VariableList, y_VariableList, testSetList, classifier, hashing=False, chi_squared=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.linear_model import RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.utils.extmath import density
    y_train = y_VariableList
    if hashing == True:
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                       n_features=2 ** 16)
        X_train = vectorizer.transform(x_VariableList)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        X_train = vectorizer.fit_transform(x_VariableList)

    X_test = vectorizer.transform(testSetList)

    if chi_squared == True:
        print("Extracting best features by a chi-squared test")
        ch2 = SelectKBest(chi2, k=2 * 16)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)

    classifierObject = ""
    print "Using :", classifier

    if classifier == "LinearSVC":
        classifierObject = LinearSVC(penalty='l2', dual=False, tol=1e-3)

    elif classifier == "PassiveAggressiveClassifier":
        classifierObject = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, loss='hinge',
                                                       n_iter=50, n_jobs=1, random_state=None, shuffle=True,
                                                       verbose=0, warm_start=False)

    elif classifier == "RidgeClassifier":
        classifierObject = RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                                           max_iter=None, normalize=False, solver='lsqr', tol=0.01)

    elif classifier == "Perceptron":
        classifierObject = Perceptron(alpha=0.0001, class_weight=None, eta0=1.0, fit_intercept=True,
                                      n_iter=50, n_jobs=1, penalty=None, random_state=0, shuffle=True,
                                      verbose=0, warm_start=False)

    elif classifier == "SGDClassifier":
        classifierObject = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
                                         eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                                         learning_rate='optimal', loss='hinge', n_iter=50, n_jobs=1,
                                         penalty='l2', power_t=0.5, random_state=None, shuffle=True,
                                         verbose=0, warm_start=False)

    classifierObject.fit(X_train, y_train)
    pred = classifierObject.predict(X_test)
    return pred[0]


def compute_emotion(mp3_file):
    trainingData = json.load(open("TrainingSet.txt"))
    trainXset = []
    trainYset = []
    testXset = [mp3_file]
    for key in trainingData.keys():
        eachClassSet = trainingData[key]
        for i in range(0, len(eachClassSet)):
            trainXset.append(eachClassSet[i])
            trainYset.append(key.split("/")[-1])
    predictions = trainOnModel(
        trainXset, trainYset, testXset, "PassiveAggressiveClassifier")
    return predictions

if __name__ == '__main__':
    process_mp3_files()
