import json
import os.path

import numpy as np
import torch
from torch.utils.data import Dataset


class MOTSynthDataset(Dataset):

    def __init__(self):
        self.motsynth_path = '/nas/softechict-nas-3/matteo/Datasets/MOTSynth/'
        self.mode = 'test'
        self.annotations_file = os.path.join(self.motsynth_path, f'motsynth_{self.mode}_tracking.json')
        self.annotations = self._load_annotations(self.annotations_file, self.mode)
        self.seq_names = list(sorted(self.annotations.keys()))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = []
        seq_name = self.seq_names[idx]
        self.sequence = self.annotations[f'{seq_name}']
        self.detections_file = f'/nas/softechict-nas-3/gmancusi/datasets/MOTSynth/yolox/sequences/yolox_bytetrack_motsynth_{seq_name}.json'
        yolox_dets = self._load_annotations(self.detections_file, self.mode)
        for key, value in self.sequence.items():
            frame_name = key
            gt = {}
            vis = {}
            im_path = os.path.join(self.motsynth_path, f'frames/{seq_name}/rgb/{frame_name}.jpg')
            dets = torch.tensor(np.array(yolox_dets[frame_name]).reshape(-1, 5)[:, :4])
            for i, detections in enumerate(value):
                gt[detections[5]] = np.array(detections[:4])
                vis[detections[5]] = float(detections[6])
            frame = {'gt': gt,
                     'im_path': im_path,
                     'vis': vis,
                     'dets': dets,
                     'seq_name': seq_name}
            data.append(frame)
        return data

    def _load_annotations(self, annotations_file, mode='train'):
        with open(annotations_file) as f:
            print(f"Loading annotations... {mode}. File: {annotations_file}. ")
            annotations = json.load(f)
        return annotations
