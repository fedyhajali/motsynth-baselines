import torch
from torch.utils.data import DataLoader
from tracktor.utils import (evaluate_mot_accums, get_mot_accum)

from motsynth_dataset import MOTSynthDataset
from utils import _load_results


def main(seq_name, results_dir, motsynth_path, dets_path, mode):
    mot_accums = []

    dataset = MOTSynthDataset(motsynth_path, dets_path, mode)
    seq_idx = dataset.seq_names.index(seq_name)
    data_loader = DataLoader(torch.utils.data.Subset(dataset[seq_idx], range(0, 1800)))

    results = _load_results(seq_name, results_dir)
    mot_accums.append(get_mot_accum(results, data_loader))

    print("Evaluation:")
    evaluate_mot_accums(mot_accums,
                        [seq_name],
                        generate_overall=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pass a sequence name')
    parser.add_argument('--seq', required=True, type=str,
                        help='the sequence name')
    parser.add_argument('--exp', required=True, type=str,
                        help='the experiment name')
    parser.add_argument('--motsynth_path', type=str,
                        default='/nas/softechict-nas-3/matteo/Datasets/MOTSynth/',
                        help='the experiment name')
    parser.add_argument('--dets_path', type=str,
                        default='/nas/softechict-nas-3/gmancusi/datasets/MOTSynth/yolox/sequences/',
                        help='the experiment name')
    parser.add_argument('--mode', type=str,
                        default='test',
                        help='the experiment name')
    args = parser.parse_args()

    results_dir = f'../output/tracktor_logs/MOTSynth/{args.exp}'
    main(args.seq, results_dir, args.motsynth_path, args.dets_path, args.mode)
