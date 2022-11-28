import csv
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torchreid.utils import FeatureExtractor
from tracktor.frcnn_fpn import FRCNN_FPN

from configs.path_cfg import OUTPUT_DIR

matplotlib.use('Agg')

# https://matplotlib.org/cycler/
# get all colors with
# colors = []
#	for name,_ in matplotlib.colors.cnames.items():
#		colors.append(name)

colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    import colorsys

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colorbar, colors
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                              boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


def plot_sequence(tracks, data_loader, output_dir, write_images):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        output_dir (String): Directory where to save the resulting images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmap = rand_cmap(len(tracks), type='bright', first_color_black=False, last_color_black=False)

    for frame_id, frame_data in enumerate(tqdm.tqdm(data_loader)):
        if frame_id == 100:
            break
        img_path = frame_data['img_path'][0]
        img = cv2.imread(img_path)[:, :, (2, 1, 0)]
        height, width, _ = img.shape

        fig = plt.figure()
        fig.set_size_inches(width / 96, height / 96)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)

        for track_id, track_data in tracks.items():
            if frame_id in track_data.keys():
                bbox = track_data[frame_id][:4]

                if 'mask' in track_data[frame_id]:
                    mask = track_data[frame_id]['mask']
                    mask = np.ma.masked_where(mask == 0.0, mask)

                    ax.imshow(mask, alpha=0.5, cmap=colors.ListedColormap([cmap(track_id)]))

                    annotate_color = 'white'
                else:
                    ax.add_patch(
                        plt.Rectangle(
                            (bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                            fill=False,
                            linewidth=2.0,
                            color=cmap(track_id)
                        ))

                    annotate_color = cmap(track_id)

                if write_images == 'debug':
                    ax.annotate(
                        f"{track_id} ({track_data[frame_id][-1]:.2f})",
                        (bbox[0] + (bbox[2] - bbox[0]) / 2.0, bbox[1] + (bbox[3] - bbox[1]) / 2.0),
                        color=annotate_color, weight='bold', fontsize=12, ha='center', va='center')

        plt.axis('off')
        # plt.tight_layout()
        plt.draw()
        plt.savefig(os.path.join(output_dir, os.path.basename(img_path)), dpi=96)
        plt.close()


def wandb_setup(cfg):
    import wandb
    import os
    from os.path import exists
    from pathlib import Path
    run_id = None
    resume = exists(Path(cfg.OUTPUT_DIR) / 'run.id')  # resume only if the log file exists
    exp_name = os.path.basename(os.path.normpath(cfg.OUTPUT_DIR))
    if resume:
        with open(Path(cfg.OUTPUT_DIR) / 'run.id', 'r') as f:
            run_id = f.readline()

    run = wandb.init(project='unbiased_teacher', entity='fedyhajali', resume=resume, id=run_id,
                     name=exp_name, allow_val_change=True, dir=f'{cfg.OUTPUT_DIR}/tmp', sync_tensorboard=True)

    # save run in the log folder
    with open(Path(cfg.OUTPUT_DIR) / 'run.id', 'w') as f:
        f.write(str(run.id))

    wandb.alert(
        title=f">>> EXP STARTED ({run.id})",
        text=f">>> Experiment: *{exp_name}* (*{run.id}*)  "
    )


def get_obj_detect_model_istance(obj_detect_model):
    if not os.path.exists(obj_detect_model):
        obj_detect_model = os.path.join(OUTPUT_DIR, 'models', obj_detect_model)
    assert os.path.isfile(obj_detect_model)
    obj_detect_state_dict = torch.load(
        os.path.join(OUTPUT_DIR, 'models', obj_detect_model), map_location=lambda storage, loc: storage)
    if 'model' in obj_detect_state_dict:
        obj_detect_state_dict = obj_detect_state_dict['model']
    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(obj_detect_state_dict, strict=False)
    obj_detect.eval()
    if torch.cuda.is_available():
        obj_detect.cuda()

    return obj_detect


def get_reid_model_istance(reid_model):
    if not os.path.exists(reid_model):
        reid_model = os.path.join(OUTPUT_DIR, 'models', reid_model)

    assert os.path.isfile(reid_model)
    reid_network = FeatureExtractor(
        model_name='resnet50_fc512',
        model_path=reid_model,
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu')

    return reid_network


def _load_results(seq_name, output_dir):
    file_path = os.path.join(output_dir, seq_name + '.txt')
    results = {}

    if not os.path.isfile(file_path):
        return results

    with open(file_path, "r") as of:
        csv_reader = csv.reader(of, delimiter=',')
        for row in csv_reader:
            frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

            if not track_id in results:
                results[track_id] = {}

            x1 = float(row[2]) - 1
            y1 = float(row[3]) - 1
            x2 = float(row[4]) - 1 + x1
            y2 = float(row[5]) - 1 + y1

            results[track_id][frame_id] = [x1, y1, x2, y2]

    return results


def write_results(all_tracks, seq_name, output_dir):
    """Write the tracks in the format for MOT16/MOT17 sumbission

     all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

     Each file contains these lines:
     <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
     """

    # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f"{seq_name}.txt"), "w") as of:
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
