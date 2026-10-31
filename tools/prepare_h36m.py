import os
import sys
import pickle
from pathlib import Path
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data.datareader_h36m import DataReaderH36M
from tqdm import tqdm


def save_clips(subset_name, root_path, train_data, train_labels):
    len_train = len(train_data)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {
            "data_input": data_input,
            "data_label": data_label
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:
            pickle.dump(data_dict, myprofile)

def main():
    parser = argparse.ArgumentParser(description='Prepare Human3.6M clips')
    parser.add_argument('--data-root', default='data/motion3d')
    parser.add_argument('--dt-file', default='h36m_sh_conf_cam_source_final.pkl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int, default=243)
    parser.add_argument('--stride', type=int, default=81)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output folder must be empty')
    reader = DataReaderH36M(n_frames=args.frames, sample_stride=1,
        data_stride_train=args.stride, data_stride_test=args.frames,
        dt_file=args.dt_file, dt_root=args.data_root)
    train, test, train_labels, test_labels = reader.get_sliced_data()
    save_clips('train', args.output, train, train_labels)
    save_clips('test', args.output, test, test_labels)

if __name__ == '__main__':
    main()
