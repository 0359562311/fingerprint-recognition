from main import FingerMatch

from libs.enhancing import *
from libs.basics import *
from libs.processing import *

path = '/home/tan/Documents/PythonProjects/AI/FingerMatch-20220508T085804Z-001/FingerMatch/data/Fingerprints - Set A'

fm = FingerMatch('tree',)
# load data image (Doc image trong folder)
fm.loadData(path)
# extract feature to class Image
fm.trainData()
# Todo: save to json
fm.save_to_json()