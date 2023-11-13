import sys
import mne
import numpy as np
from random import randint


"""
This script allows the user to generate a randomized training set (or test set)
from the eegmmidb dataset. Choose number of samples and fragment length as arguments.

After the training set generation execute generate-tES with apropriate parameters
to generate the synthetic estimulation artifacts for the generated set.
"""
#https://www.physionet.org/content/eegmmidb/1.0.0/#files-panel
trainsetPath = "data/EEG_all_epochs.npy"
dataBaseFolder = "MotorDataset/"
sfreq = 160

if __name__ == "__main__":
	entityNum = int(sys.argv[1]) #Number of entities
	entityLength = int(sys.argv[2]) * sfreq #Length of entities in seconds
	trainset = np.empty([entityNum, entityLength])

	for i in range(entityNum):
		randParticipant = randint(1, 109)
		randScan = randint(1, 14)

		participantFolder = "S" + str(randParticipant).zfill(3)
		scanFilename = participantFolder + "R" + str(randScan).zfill(2) + ".edf"
		edfPath = dataBaseFolder + participantFolder + "/" + scanFilename

		data = mne.io.read_raw_edf(edfPath, verbose="ERROR")
		randtime = randint(0, data.n_times-entityLength-1)
		randchannel = data.ch_names[randint(0, 63)]
		rawdata = data.get_data(picks=[randchannel], start=randtime, stop=randtime+entityLength)
		trainset[i] = rawdata[0]

		print(str(i+1)+"/"+str(entityNum)+": "+edfPath+"\nChannel:"+randchannel+" # Startpoint:"+str(randtime))

print(">>>Writing training set to disk<<<")
trainset *= 1e6
np.save(trainsetPath, trainset)
