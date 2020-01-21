#!/usr/bin/env python3

import random
import numpy as np
import statistics

class Ensemble:
    def __init__(self, networks, num_classes):
        self.__num_classes = num_classes
        self.__networks = networks
        self.__possible_uncertainties = []
        uncertainty = 0
        while uncertainty <= round(1 - (1 / self.__num_classes), 2):
            self.__possible_uncertainties.append(uncertainty)
            uncertainty = round(uncertainty + (1 / len(self.__networks)), 2)
    
    def train(self, train_x, train_y, batch_size, steps_per_epoch, num_epochs, valid_x, valid_y):
        for network in self.__networks:
            network.fit(
                x=data_generator(train_x, train_y, batch_size=batch_size),
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs,
                validation_data=data_generator(valid_x, valid_y, batch_size=batch_size),
                validation_steps=1
            )
            
    def load(self, file_front, file_end, valid_x, valid_y):
        for network in self.__networks:
            network.load_weights(file_front + str(self.__networks.index(network)) + file_end)
            
    def getUncertaintyCorrectness(self, x, y):
        uncertainties, ensemble_predictions = self.getUncertaintiesAndEnsemblePredictions(x)
        erg = []
        correct = []
        count = []
        for i in range(len(self.__possible_uncertainties)):
            correct.append(0)
            count.append(0)
        for i in range(len(ensemble_predictions)):
            index = self.__possible_uncertainties.index(uncertainties[i])
            count[index] += 1
            if ensemble_predictions[i] == y[i]:
                correct[index] += 1
        for i in range(len(correct)):
            if correct[i] > 0:
                erg.append(1 - correct[i] / count[i])
            else:
                erg.append(-1)
        return erg
            
    def getPossibleUncertainties(self):
        return self.__possible_uncertainties
    
    def getEnsemblePredictions(self, x):
        return self.__getEnsemblePredictions(self.getPredictions(x))
    
    def getUncertainties(self, x):
        return self.__getUncertainties(self.getPredictions(x))
    
    def getUncertaintiesAndEnsemblePredictions(self, x):
        predictions = self.getPredictions(x)
        return self.__getUncertainties(predictions), self.__getEnsemblePredictions(predictions)
    
    def getPredictions(self, x):
        predictions = []
        for network in self.__networks:
            predictions.append(network.predict_classes(x))
        return predictions
    
    def getAccuracies(self, valid_x, valid_y):
        accuracies = []
        for network in self.__networks:
            accuracies.append(self.__getAccuracy(network, valid_x, valid_y))
        return accuracies
    
    def getEnsembleAccuracy(self, valid_x, valid_y):
        successes = 0
        predictions = self.getPredictions(valid_x)
        ensemble_predictions = self.__getEnsemblePredictions(predictions)
        for i in range(len(ensemble_predictions)):
            if ensemble_predictions[i] == valid_y[i]:
                successes += 1
        return successes / len(ensemble_predictions)
    
    def getNetworks(self):
        return self.__networks
        
    def save(self, file_front, file_end):
        for network in self.__networks:
            network.save_weights(file_front + str(self.__networks.index(network)) + file_end)
            
    def __getAccuracy(self, network, valid_x, valid_y):
        x = network.predict_classes(valid_x)
        successes = 0
        for i in range(len(x)):
            if x[i] == valid_y[i]:
                successes += 1
        return successes / len(x)

    def __getUncertainties(self, predictions):
        uncertainties = []
        temp = []
        for i in range(len(predictions[0])):
            for prediction in predictions:
                temp.append(prediction[i])
            predicted = self.__getMostOccured(temp, self.__num_classes)
            uncertainties.append(round(1 - (temp.count(predicted) / len(temp)),2))
            temp.clear()
        return uncertainties
            
    def __getEnsemblePredictions(self, predictions):
        ensemble_predictions = []
        temp = []
        for i in range(len(predictions[0])):
            # Alle Werte einer Ebene in eine Liste
            for prediction in predictions:
                temp.append(prediction[i])
            # Aus dieser Liste den häufigsten Wert als Ergebnis wählen
            ensemble_prediction = self.__getMostOccured(temp, self.__num_classes)
            ensemble_predictions.append(ensemble_prediction)
            temp.clear()
        return ensemble_predictions
            
    def __getMostOccured(self, values, possible_values):
        temp = []
        for i in range(possible_values):
            temp.append(values.count(i))
        return temp.index(max(temp))
    
# Aus https://www.curious-containers.cc/docs/machine-learning-guide entnommen (gesehen 18.01.2020)
def data_generator(x, y, batch_size=None):
    index = range(len(x))
    labels = np.array([[1, 0], [0, 1]])

    while True:
        index_sample = index
        if batch_size is not None:
            index_sample = sorted(random.sample(index, batch_size))

        x_data = x[index_sample] / 256.0
        y_data = y[index_sample]
        y_data = labels[y_data[:, 0, 0, 0]]
        yield x_data, y_data