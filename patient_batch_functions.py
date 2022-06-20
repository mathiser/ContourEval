import os
import json
from . import Patient

def batch_patient_load(dir, label_dict_path):
    with open(label_dict_path, "r") as r:
        label_dict = json.loads(r.read())
    
    label_dict = {int(k): v for k, v in label_dict.items()}
    
    d = {}
    for f in os.listdir(dir):
        if not f.endswith("nii.gz"):
            continue
        full_path = os.path.join(dir, f)

        p = Patient(full_path, label_dict)
        d[p.id] = p
    return d

def batch_dice(gt_d: dict, pred_d: dict):
    for k, v in gt_d.items():
        pass