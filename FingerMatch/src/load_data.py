from main import FingerMatch

from libs.enhancing import *
from libs.basics import *
from libs.processing import *

path = '/home/tan/Documents/PythonProjects/AI/FingerMatch-20220508T085804Z-001/FingerMatch/data/Fingerprints - Set A'

fm = FingerMatch('tree',)
# load data
fm.loadData(path)
# extract features
fm.trainData()
# save
fm.save_as_pickle()

print(True.sum(1))