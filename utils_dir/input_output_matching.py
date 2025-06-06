import os


def get_x_y_no(input_dir):
    files = os.listdir(input_dir)
    files = [f for f in files if f.endswith('.tif')]
    x_set = list()
    y_set = list()
    for f in files:
        x_y = f.split('_')
        x_set.append(int(x_y[1]))
        y_set.append(int(x_y[2]))

    return max(x_set) + 1, max(y_set) + 1


def input_output_matching(input_dir, output_dir, y_no, index):
    files = os.listdir(input_dir)
    files = [f for f in files if f.endswith('.tif')]
    file = files[index]
    x_y = file.split('_')
    tile_no = int(x_y[1]) * (y_no) + int(x_y[2])
    output_file = os.path.join(output_dir, "tile_" + str(tile_no) + '.tif')

    return output_file


if __name__ == '__main__':
    x_no, y_no = get_x_y_no(r'D:\data\20230913\raw_data\mouse_brain_s260_f260_z_step_2_1')

    input_output_matching(
        input_dir=r'D:\data\20230913\raw_data\mouse_brain_s260_f260_z_step_2_1',
        output_dir=r'D:\data\20230913\raw_data\mouse_brain_s260_f260_z_step_2_1_merged',
        y_no=y_no,
        index=3299
    )