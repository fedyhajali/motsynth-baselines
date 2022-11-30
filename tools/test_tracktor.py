import os
import sys
import time

import motmetrics as mm
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from tracktor.datasets.factory import Datasets
from tracktor.tracker import Tracker
from tracktor.utils import (evaluate_mot_accums, get_mot_accum, interpolate_tracks)

from motsynth_dataset import MOTSynthDataset
from utils import (plot_sequence, get_obj_detect_model_istance, get_reid_model_istance, _load_results, write_results)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from configs.path_cfg import OUTPUT_DIR

mm.lap.default_solver = 'lap'


def main(module_name, name, seed, obj_detect_models, reid_models,
         tracker, dataset, frame_range, interpolate,
         write_images, load_results, motsynth_root, dets_root, mode):
    # set all seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    output_dir = os.path.join(OUTPUT_DIR, 'tracktor_logs', module_name, name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset == 'MOTSynth':
        dataset = MOTSynthDataset(motsynth_root, dets_root, mode)
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
        tracker.reset()

        print(f"Tracking: {seq}")

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))

        data_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))
        num_frames += len(data_loader)
        seq_name = str(seq[0]['seq_name'])
        eval_seqs.append(seq_name)

        results = {}
        if load_results:
            results = _load_results(seq_name, output_dir)
        if not results:
            start = time.time()

            for frame_data in tqdm(data_loader):
                with torch.no_grad():
                    # image generation here to lighten the dataloader, we couldn't load all images for entire sequence
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

        mot_accums.append(get_mot_accum(results, data_loader))

        if write_images:
            plot_sequence(
                results,
                data_loader,
                os.path.join(output_dir, str(dataset), str(seq_name)),
                write_images)

    if time_total:
        print(f"Tracking runtime for all sequences (without evaluation or image writing): "
              f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        print("Evaluation:")
        evaluate_mot_accums(mot_accums,
                            [str(seq[0]['seq_name']) for seq in dataset],
                            generate_overall=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pass a .yaml conf filename')
    parser.add_argument('--conf', required=True,
                        help='the configuration filename')
    args = parser.parse_args()
    with open(f'../configs/{args.conf}.yaml', 'r') as file:
        args = yaml.safe_load(file)

    main(args['module_name'], args['name'], args['seed'], args['obj_detect_models'],
         args['reid_models'], args['tracker'], args['dataset'], args['frame_range'],
         args['interpolate'], args['write_images'], args['load_results'],
         args['motsynth_root'], args['dets_root'], args['mode'])
