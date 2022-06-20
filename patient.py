import os

import SimpleITK as sitk
import numpy as np
class Patient:
    def __init__(self, nifti_path, label_dict=None):
        self.nifti_path = nifti_path
        self.id = os.path.basename(self.nifti_path.split("&")[0])
        self.contours_image = None
        self.contours_array = None
        self.uniques = None
        self.label_dict = label_dict

    def get_oar_image_by_int(self, i: int) -> sitk.Image:
        return self.contours_image == i

    def as_image(self):
        if self.contours_image is None:
            self.contours_image = sitk.ReadImage(self.nifti_path)

        return self.contours_image

    def as_array(self):
        if self.contours_array is None:
            self.contours_array = sitk.GetArrayFromImage(self.as_image())

        return self.contours_array

    def get_unique_contour_ints(self):
        if self.uniques is None:
            self.uniques = np.unique(self.as_array())

        return self.uniques