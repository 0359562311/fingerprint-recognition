from main import FingerMatch

from libs.enhancing import *
from libs.basics import *
from libs.processing import *

path = '/home/hoangdo/Documents/python/fingerprint-recognition/FingerMatch/data/Fingerprints - Set A'
fm = FingerMatch('tree',)
fm.load_from_json()
# image image -> matching 
input_image = load_image('/home/hoangdo/Documents/python/fingerprint-recognition/FingerMatch/data/Fingerprints - Set B/101_1.tif', True)
print(fm.matchFingerprint(input_image,verbose=False))