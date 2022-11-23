import csv
import sys
import time
from os import path as osp

import motmetrics as mm
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchreid.utils import FeatureExtractor
from torchvision.transforms import ToTensor
from tqdm import tqdm
from tracktor.datasets.factory import Datasets
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.tracker import Tracker
from tracktor.utils import (evaluate_mot_accums, get_mot_accum,
                            interpolate_tracks, plot_sequence)

from motsynth_dataset import MOTSynthDataset

sys.path.append(osp.dirname(osp.dirname(__file__)))
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)), 'src'))
from configs.path_cfg import MOTCHA_ROOT, OUTPUT_DIR

from tracktor.config import cfg as _cfg

import os

if not osp.exists(_cfg.DATA_DIR):
    os.symlink(MOTCHA_ROOT, _cfg.DATA_DIR)

mm.lap.default_solver = 'lap'


def get_obj_detect_model_istance(obj_detect_model):
    if not osp.exists(obj_detect_model):
        obj_detect_model = osp.join(OUTPUT_DIR, 'models', obj_detect_model)
    assert os.path.isfile(obj_detect_model)
    obj_detect_state_dict = torch.load(
        osp.join(OUTPUT_DIR, 'models', obj_detect_model), map_location=lambda storage, loc: storage)
    if 'model' in obj_detect_state_dict:
        obj_detect_state_dict = obj_detect_state_dict['model']
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(obj_detect_state_dict, strict=False)
    obj_detect.eval()
    if torch.cuda.is_available():
        obj_detect.cuda()

    return obj_detect


def get_reid_model_istance(reid_model):
    if not osp.exists(reid_model):
        reid_model = osp.join(OUTPUT_DIR, 'models', reid_model)

    assert os.path.isfile(reid_model)
    reid_network = FeatureExtractor(
        model_name='resnet50_fc512',
        model_path=reid_model,
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu')

    return reid_network


def write_results(all_tracks, seq_name, output_dir):
    """Write the tracks in the format for MOT16/MOT17 sumbission

     all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

     Each file contains these lines:
     <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
     """

    # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(osp.join(output_dir, f"{seq_name}.txt"), "w") as of:
        writer = csv.writer(of, delimiter=',')
        for i, track in all_tracks.items():
            for frame, bb in track.items():
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                writer.writerow(
                    [frame + 1,
                     i + 1,
                     x1 + 1,
                     y1 + 1,
                     x2 - x1 + 1,
                     y2 - y1 + 1,
                     -1, -1, -1, -1])


def main(module_name, name, seed, obj_detect_models, reid_models,
         tracker, dataset, test_motsynth, frame_range, interpolate,
         write_images):
    # set all seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(OUTPUT_DIR, 'tracktor_logs', module_name, name)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    if test_motsynth:
        dataset = MOTSynthDataset()
    else:
        dataset = Datasets(dataset)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    print("Initializing object detector.")
    obj_detect = get_obj_detect_model_istance(obj_detect_models)
    # reid
    print("Initializing reID network.")
    reid_network = get_reid_model_istance(reid_models)
    # tracktor
    print("Initializing Tracker.")
    tracker = Tracker(obj_detect, reid_network, tracker)
    tracker.obj_detect = obj_detect
    tracker.reid_network = reid_network

    time_total = 0
    num_frames = 0
    mot_accums = []
    eval_seqs = []

    for seq in dataset:
        # Messy way to evaluate on MOTS without having to modify code from the tracktor repo
        eval_seqs.append(str(seq))
        tracker.reset()

        print(f"Tracking: {seq}")

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))

        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))
        num_frames += len(seq_loader)
        seq_name = None

        results = {}
        if not results:
            start = time.time()

            for frame_data in tqdm(seq_loader):
                with torch.no_grad():
                    if not seq_name:
                        seq_name = frame_data['seq_name']
                    frame_data['img'] = ToTensor()(Image.open(frame_data['im_path'][0]).convert("RGB")).unsqueeze(0)
                    tracker.step(frame_data)

            results = tracker.get_results()

            time_total += time.time() - start

            print(f"Tracks found: {len(results)}")
            print(f"Runtime for {seq}: {time.time() - start :.2f} s.")

            if interpolate:
                results = interpolate_tracks(results)

            print(f"Writing predictions to: {output_dir}")
            write_results(results, seq_name, output_dir)

        if seq.no_gt:
            print("No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq_loader))

        if write_images:
            plot_sequence(
                results,
                seq,
                osp.join(output_dir, str(dataset), str(seq)),
                write_images)

    if time_total:
        print(f"Tracking runtime for all sequences (without evaluation or image writing): "
              f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        print("Evaluation:")
        evaluate_mot_accums(mot_accums,
                            [str(s) for s in dataset if not s.no_gt and str(s) in eval_seqs],
                            generate_overall=True)


if __name__ == "__main__":
    with open('configs/tracktor.yaml', 'r') as file:
        args = yaml.safe_load(file)

    main(args['module_name'], args['name'], args['seed'], args['obj_detect_models'], args['reid_models'],
         args['tracker'], args['dataset'], args['test_motsynth'], args['frame_range'], args['interpolate'],
         args['write_images'])
