# Unsicherheitsquantifizierung von Tumordetektion
Der Code zu der Bachelorarbeit "Die Unsicherheitsquantifizierung einer KI-basierten Tumordetektion" von Ottokar Elgeti an der HTW Berlin.  
Die 'Ensemble'-Klasse befindet sich in der Datei 'ensemble.py'.  
Das ausgeführte Programm befindet sich in der Datei 'uncertainty.py'.  
Die gespeicherten Gewichte der Neuronalen Netzwerke befinden sich im 'weights'-Ordner.  
Die Metriken der Ensembles befinden sich im 'metrics'-Ordner.  
Innerhalb des 'metrics'-Ordners gibt es einen 'plots'-Ordner, der die Graphen für jedes Ensemble enthält.  
Die Datei 'uncertainty.red.json' wird genutzt um das Programm mittels CuriousContainers auszuführen.  
  
  
Ist Git installiert kann der Code, zusammen mit den Metriken, Gewichten und Plots mit dem Befehl 'git clone https://github.com/OElgeti/unsicherheitsquantifizierung\_tumordetektion.git' heruntergeladen werden. Dieser Befehl erstellt an der derzeitigen Stelle den Ordner 'unsicherheitsquantifizierung\_tumordetektion'.  
Zur Installation der benötigten Software sollte dem Tutorial unter 'https://www.curious-containers.cc/docs/red-beginners-guide' bis zu dem Punkt 'Sample Application' gefolgt werden. Wenn Zugang zum nicht-öffentlichen HTW-Server 'avocado01.f4.htw-berlin.de' und der Agency 'https://agency.f4.htw-berlin.de/dt' besteht, muss nur noch der Befehl 'faice exec uncertainty.red.json' ausgeführt werden.  
Sind diese Dinge nicht gegeben, können bei 'uncertainty.red.json' in Zeile 171 die Agency und in den Zeilen 92, 110, 128 und 142 der Server geändert werden. Wird ein eigener Server verwendet, sollte dem Tutorial unter 'https://www.curious-containers.cc/docs/machine-learning-guide' bis zum Abschnitt 'Training Experiment' gefolgt werden.  
DER FOLGENDE TEIL KONNTE AUFGRUND BESCHRÄNKTER HARDWARE NICHT GETESTET WERDEN!!!  
Soll das Programm nur lokal laufen, kann der Inhalt von 'execution' in 'uncertainty.red.json' durch:  
    "execution": {  
	"engine": "ccfaice",  
	"settings": {}  
    }  
ersetzt werden. Zu beachten ist, dass in diesem Fall die Pfade für das lokale System angepasst werden, vor allem 'dirPath' in Zeile 97.  
