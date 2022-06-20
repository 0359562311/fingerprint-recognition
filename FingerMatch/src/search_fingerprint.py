from main import FingerMatch

from libs.enhancing import *
from libs.basics import *
from libs.processing import *

path = '/home/tan/Documents/PythonProjects/AI/FingerMatch-20220508T085804Z-001/FingerMatch/data/Fingerprints - Set A'

fm = FingerMatch('tree',)
fm.load_from_pickle()

input_image = load_image('/home/tan/Documents/PythonProjects/AI/FingerMatch-20220508T085804Z-001/FingerMatch/data/Fingerprints - Set B/101_1.tif', True)
print(fm.matchFingerprint(input_image,verbose=False))