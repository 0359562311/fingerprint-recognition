# Class build
import pickle
import json
import re
import os
import glob
import time
import operator

import numpy as np

from libs.minutiae import minutiae_points, plot_minutiae, process_minutiae, generate_tuple_profile
from libs.matching import match_tuples, evaluate
from libs.edges import match_edge_descriptors
from libs.basics import load_image
from libs.enhancing import enhance_image
from libs.edges import edge_processing, sift_match


class Image:
    """
    Containing element for images - stores image array and its tuple profile.

    """
    def __init__(self, img_id: str, path: str, image_raw: np.array, image_enhanced: np.array, profile: dict):
        self.img_id = img_id
        self.path = path
        self.image_raw = image_raw
        self.image_enhanced = image_enhanced
        self.minutiae = None
        self.profile = profile

    def plot(self):
        """
        Plots minutiae from the stored image.

        """

        plot_minutiae(self.image_enhanced, list(self.profile.keys()), size=8)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
class FingerMatch:
    def __init__(self, model: str = 'tree', threshold: int = 125):
        self.images = []
        self.model = model
        self.threshold = threshold

    def loadData(self, path: str, image_format: str = 'tif', limit: int = None) -> None:
        """
        Load data that matches the image_format, from the given path. Each image is processed and stored.

        """

        # img_paths = [glob.glob(f'{path}/*.{image_format}', recursive=True)][0]
        img_paths = [path + '/' + f for f in os.listdir(path)]

        try:
            assert len(img_paths) > 0
        except:
            raise FileNotFoundError(f'ERROR: No image files available to extract from the path {path}')

        if limit is not None:
            # Restrict sample size.
            img_paths = img_paths[:limit]

        start = time.time()

        for p in img_paths:
            # Image loading
            image_raw = load_image(p, True)

            try:
                # Image properties definition.
                img_id = re.search(f'(.+?).{image_format}', os.path.basename(p)).group(1)
            except AttributeError:
                raise Exception(f'ERROR: Unknown image id for {p}')

            # Create new profile for the given image and store it.
            self.images.append(Image(img_id, p, image_raw, None, None))

        print(f'\nINFO: Dataset loaded successfully. Duration: {round(time.time() - start, 2)} sec')

    def trainData(self):
        """
        Loads model on the given dataset.

        """

        start = time.time()
        print(f'INFO: Loading model features. Model: {self.model.lower()}')

        if self.model.lower() == 'tree':
            for i in range(len(self.images)):
                # Extract minutiae.
                try:
                    self.images[i].image_enhanced = enhance_image(self.images[i].image_raw, skeletonise=True)
                    minutiae = process_minutiae(self.images[i].image_enhanced)

                    # Confirmed point matching.
                    self.images[i].profile = generate_tuple_profile(minutiae)
                    
                    # Rewriting to the loaded data.
                    self.images[i].minutiae = minutiae
                except:
                    pass

        print(f'INFO: Training completed in {round(time.time() - start, 2)} sec')

    def save_as_pickle(self):
        with open("/home/tan/Documents/PythonProjects/AI/FingerMatch-20220508T085804Z-001/FingerMatch/src/dt.json", "wb") as output:
            pickle.dump(self.images, output)

    def save_to_json(self):
        ar = []
        images = self.images
        for i in images:
            profileDict = {}
            for x, y in i.profile.items():
                profileDict[str(x)] = y
            i.profile = profileDict
            i.image_raw = None
            i.path = None
            i.image_enhanced = None
            i.minutiae = None
            # print(json.dumps(i.__dict__, cls=NumpyEncoder))
            ar.append(json.dumps(i.__dict__, cls=NumpyEncoder))
        with open("/home/tan/Documents/PythonProjects/AI/FingerMatch-20220508T085804Z-001/FingerMatch/src/dt.json", "w") as op:
            json.dump(ar, op)

    def load_from_pickle(self):
        with open("/home/hoangdo/Documents/python/fingerprint-recognition/FingerMatch/src/data.json", "rb") as f:
            self.images = pickle.load(f)

    def load_from_json(self):
        with open("/home/tan/Documents/PythonProjects/AI/FingerMatch-20220508T085804Z-001/FingerMatch/src/dt.json", "r") as f:
            images = json.load(f)
            minutiae = []
            for i in images:
                profileDict = {}
                img = json.loads(i)
                for x, y in img["profile"].items():
                    profileDict[tuple(x)] = y
                    minutiae.append(tuple(x))
                img["profile"] = profileDict
                img["minutiae"] = minutiae
                self.images.append(img)

    def matchFingerprint(self, image: np.array, verbose: bool = False, match_th: int = 33):
        """
        The given image is compared against the loaded templates.
        A similarity score is computed and used to determine the most likely match, if any.

        """
        if self.model.lower() == 'tree':

            img_test = enhance_image(image, skeletonise=True) # image input enhance

            minutiae_test = process_minutiae(img_test) ## all minutiae 
            # Confirmed point matching.
            img_profile = generate_tuple_profile(minutiae_test) # all profile

            matchest_fingerprint = self.images[0]
            match_point = 0

            for i in range(len(self.images)):
                # Matching.
                # So diem minutiae trung nhau giua 2 anh
                common_points_base, common_points_test = match_tuples(self.images[i]["profile"], img_profile)

                # So diem minutiae max giua 2 anh(anh dau vao va anh dang truy van) 
                minutiae_score = max(len(self.images[i]["profile"]), len(img_profile), 1)

                if len(common_points_base) / minutiae_score > match_point:
                    match_point = len(common_points_base) / minutiae_score #So diem trung nhau cua anh dau vao va anh truy van / max
                    matchest_fingerprint = self.images[i]

            print(f'Matching fingerprint is {matchest_fingerprint["img_id"]} with points: {match_point}')
