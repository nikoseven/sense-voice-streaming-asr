import numpy as np


def load_cmvn(cmvn_file):
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if not line_item:
            continue
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue

    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    return np.array([means, vars])


def apply_cmvn(inputs, cmvn):
    frame, dim = inputs.shape
    means = cmvn[0:1, :dim]  # Shape: (1, dim)
    vars = cmvn[1:2, :dim]  # Shape: (1, dim)
    return (inputs + means) * vars
