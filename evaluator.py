from multiprocessing.pool import ThreadPool

import SimpleITK as sitk
import numpy as np
from numba import njit
from .patient import Patient
import pandas as pd
import json

def calculate_hausdorff(gt_patient: Patient, pred_patient: Patient):
    intersection = tuple(np.intersect1d(gt_patient.get_unique_contour_ints(), pred_patient.get_unique_contour_ints()))

    def execute(gt_image: sitk.Image, pred_image: sitk.Image, label_int: int):
        filter = sitk.HausdorffDistanceImageFilter()
        filter.Execute(gt_image == label_int, pred_image == label_int)
        return label_int, filter.GetHausdorffDistance()

    t = ThreadPool(8)
    results = t.starmap(execute, [(gt_patient.as_image(), pred_patient.as_image(), i) for i in intersection])
    t.close()
    t.join()

    results_dict = {r[0]: r[1] for r in results}
    return {gt_patient.id: results_dict}

def calculate_average_hausdorff(gt_patient: Patient, pred_patient: Patient):
    intersection = tuple(np.intersect1d(gt_patient.get_unique_contour_ints(), pred_patient.get_unique_contour_ints()))

    def execute(gt_image: sitk.Image, pred_image: sitk.Image, label_int: int):
        filter = sitk.HausdorffDistanceImageFilter()
        filter.Execute(gt_image == label_int, pred_image == label_int)
        return label_int, filter.GetAverageHausdorffDistance()

    t = ThreadPool(1)
    results = t.starmap(execute, [(gt_patient.as_image(), pred_patient.as_image(), i) for i in intersection])
    t.close()
    t.join()

    results_dict = {r[0]: r[1] for r in results}
    return {gt_patient.id: results_dict}

def calculate_dice(gt_patient: Patient, pred_patient: Patient):
    intersection = tuple(np.intersect1d(gt_patient.get_unique_contour_ints(), pred_patient.get_unique_contour_ints()))

    def execute(gt_image: sitk.Image, pred_image: sitk.Image, label_int: int):
        filter = sitk.LabelOverlapMeasuresImageFilter()

        filter.Execute(gt_image == label_int, pred_image == label_int)
        return label_int, filter.GetDiceCoefficient()

    t = ThreadPool(8)
    results = t.starmap(execute, [(gt_patient.as_image(), pred_patient.as_image(), i) for i in intersection])
    t.close()
    t.join()
    results_dict = {r[0]: r[1] for r in results}
    return {gt_patient.id: results_dict}

@njit
def get_edge_of_mask(mask: np.ndarray) -> np.ndarray:
    """
    Mask must only contain 0 for background and 1 for mask.
    :param mask:
    :return:
    """
    edge = np.zeros_like(mask)
    for z in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for x in range(0, mask.shape[2]):
                sum = np.sum(mask[z, y-1:y+2, x-1:x+2])
                if sum < 9:
                    edge[z, y, x] = mask[z, y, x]
    return edge

def calculate_added_path_length(gt_patient: Patient, pred_patient: Patient):
    def execute(gt_image: sitk.Image, pred_image: sitk.Image, label_int: int):
        gt_edge = get_edge_of_mask(sitk.GetArrayFromImage(gt_image == label_int))
        pred_edge = get_edge_of_mask(sitk.GetArrayFromImage(pred_image  == label_int))
            
        ## Edge case if prediction is all false and should not be. If so, return full size of prediction
        if np.count_nonzero(gt_edge) == 0:
            apl = np.count_nonzero(pred_edge)
        else:
            apl = (gt_edge > pred_edge).astype(int).sum()
    
        return label_int, apl
    
    union = np.union1d(gt_patient.get_unique_contour_ints(), pred_patient.get_unique_contour_ints())
    #print(list(union))
    results = []
    for i in union:
        results.append(execute(gt_patient.as_image(), pred_patient.as_image(), i))

    results_dict = {r[0]: r[1] for r in results}
    return {gt_patient.id: results_dict}

def calculate_added_path_ratio(gt_patient: Patient, pred_patient: Patient):
    intersection = np.intersect1d(gt_patient.get_unique_contour_ints(), pred_patient.get_unique_contour_ints())

    def execute(gt_image: sitk.Image, pred_image: sitk.Image, label_int: int):
        gt_edge = get_edge_of_mask(sitk.GetArrayFromImage(gt_image == label_int))
        pred_edge = get_edge_of_mask(sitk.GetArrayFromImage(pred_image == label_int))
        diff = gt_edge - pred_edge
        count = np.count_nonzero(diff)
        ratio = count / np.count_nonzero(gt_edge)
        return label_int, ratio

    results = []
    for i in intersection:
        results.append(execute(gt_patient.as_image(), pred_patient.as_image(), i))

    results_dict = {r[0]: r[1] for r in results}
    return {gt_patient.id: results_dict}

def generate_metric_csv(gt_dict, pred_dict, metric_function, label_dict_path, output_path):
    
    label_dict = load_label_dict(label_dict_path)
    
    df = pd.DataFrame()
    fails = []
    for k, v in gt_dict.items():
        #print(f"Calculating for {k}")
        try:
            new_row_dict = metric_function(gt_dict[k], pred_dict[k])
            new_row_df = pd.DataFrame.from_dict(new_row_dict, orient="index")
            df = pd.concat([df, new_row_df])
        except Exception as e:
            print(e)
            print(gt_dict[k].as_image().GetSize(), pred_dict[k].as_image().GetSize())
            fails.append(k)
    

    with open(output_path.replace(".csv", ".failed.csv"), "w") as f:
        print(fails)
        f.write(json.dumps(fails))
    
    df.rename(columns = label_dict, inplace = True)
    df.to_csv(output_path)
    return df.sort_index()

def load_label_dict(label_dict_path):
    with open(label_dict_path, "r") as r:
        label_dict = json.loads(r.read())
    
    return {int(k): v for k, v in label_dict.items()}

def generate_csv_for_bounds(bounds_dict, label_dict_path, output_path):
    
    label_dict = load_label_dict(label_dict_path)
    
    df = pd.DataFrame()
    fails = []
    new_row_dict = {}
    for k, v in bounds_dict.items():
        print(f"Calculating for {k}")
        new_row_dict[k] = {}
        for i in v.get_unique_contour_ints():
            label_img = bounds_dict[k].get_oar_image_by_int(i)
            label_arr = sitk.GetArrayFromImage(label_img)
            new_row_dict[k][i] = np.count_nonzero(get_edge_of_mask(label_arr))
        

        
    new_row_df = pd.DataFrame.from_dict(new_row_dict, orient="index")
    df = pd.concat([df, new_row_df])
    
    df.rename(columns = label_dict, inplace = True)
    df.to_csv(output_path)
    return df.sort_index()