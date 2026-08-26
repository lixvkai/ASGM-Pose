import numpy as np
import copy

def flip_data(data):
    """
    horizontal flip
        data: [N, F, 17, D] or [F, 17, D]. X (horizontal coordinate) is the first channel in D.
    Return
        result: same
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints+right_joints, :] = flipped_data[..., right_joints+left_joints, :]
    return flipped_data

def resample(ori_len, target_len, replay=False, randomness=True):
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len-target_len)
            return range(st, st+target_len)
        else:
            return np.array(range(target_len)) % ori_len
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel*low+(1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape)*interval + even
            result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result

def split_clips(vid_list, n_frames, data_stride):
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i<len(vid_list):
        i += 1
        if i-st == n_frames:
            result.append(range(st,i))
            saved.add(vid_list[i-1])
            st = st + data_stride
            n_clips += 1
        if i==len(vid_list):
            break
        if vid_list[i]!=vid_list[i-1]:
            if not (vid_list[i-1] in saved):
                resampled = resample(i-st, n_frames) + st
                result.append(resampled)
                saved.add(vid_list[i-1])
            st = i
    return result