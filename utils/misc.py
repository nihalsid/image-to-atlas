import inspect

import torch
from PIL import Image
import numpy as np
from pathlib import Path
from moviepy.editor import ImageSequenceClip


def get_all_4_locations(values_y, values_x):
    y_0 = torch.floor(values_y)
    y_1 = torch.ceil(values_y)
    x_0 = torch.floor(values_x)
    x_1 = torch.ceil(values_x)
    return torch.cat([y_0, y_0, y_1, y_1], 0).long(), torch.cat([x_0, x_1, x_0, x_1], 0).long()


def concat_pil_images(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_image.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_image


def tensor1F_to_PIL3U(arr):
    return Image.fromarray((arr.expand(-1, -1, 3).detach().cpu().numpy() * 255).astype(np.uint8))


def tensor3F_to_PIL3U(arr):
    return Image.fromarray((arr.detach().cpu().numpy() * 255).astype(np.uint8))


def printarr(*arrs, float_width=6):
    """
    Print a pretty table giving name, shape, dtype, type, and content information for input tensors or scalars.

    Call like: printarr(my_arr, some_other_arr, maybe_a_scalar). Accepts a variable number of arguments.

    Inputs can be:
        - Numpy tensor arrays
        - Pytorch tensor arrays
        - Jax tensor arrays
        - Python ints / floats
        - None

    It may also work with other array-like types, but they have not been tested.

    Use the `float_width` option specify the precision to which floating point types are printed.

    Author: Nicholas Sharp (nmwsharp.com)
    Canonical source: https://gist.github.com/nmwsharp/54d04af87872a4988809f128e1a1d233
    License: This snippet may be used under an MIT license, and it is also released into the public domain.
             Please retain this docstring as a reference.
    """

    frame = inspect.currentframe().f_back
    default_name = "[temporary]"

    ## helpers to gather data about each array
    def name_from_outer_scope(a):
        if a is None:
            return '[None]'
        name = default_name
        for k, v in frame.f_locals.items():
            if v is a:
                name = k
                break
        return name

    def dtype_str(a):
        if a is None:
            return 'None'
        if isinstance(a, int):
            return 'int'
        if isinstance(a, float):
            return 'float'
        return str(a.dtype)

    def shape_str(a):
        if a is None:
            return 'N/A'
        if isinstance(a, int):
            return 'scalar'
        if isinstance(a, float):
            return 'scalar'
        return str(list(a.shape))

    def type_str(a):
        return str(type(a))[8:-2]  # TODO this is is weird... what's the better way?

    def device_str(a):
        if hasattr(a, 'device'):
            device_str = str(a.device)
            if len(device_str) < 10:
                # heuristic: jax returns some goofy long string we don't want, ignore it
                return device_str
        return ""

    def format_float(x):
        return f"{x:{float_width}g}"

    def minmaxmean_str(a):
        if a is None:
            return ('N/A', 'N/A', 'N/A')
        if isinstance(a, int) or isinstance(a, float):
            return (format_float(a), format_float(a), format_float(a))

        # compute min/max/mean. if anything goes wrong, just print 'N/A'
        min_str = "N/A"
        try:
            min_str = format_float(a.min())
        except:
            pass
        max_str = "N/A"
        try:
            max_str = format_float(a.max())
        except:
            pass
        mean_str = "N/A"
        try:
            mean_str = format_float(a.mean())
        except:
            pass

        return (min_str, max_str, mean_str)

    try:

        props = ['name', 'dtype', 'shape', 'type', 'device', 'min', 'max', 'mean']

        # precompute all of the properties for each input
        str_props = []
        for a in arrs:
            minmaxmean = minmaxmean_str(a)
            str_props.append({
                'name': name_from_outer_scope(a),
                'dtype': dtype_str(a),
                'shape': shape_str(a),
                'type': type_str(a),
                'device': device_str(a),
                'min': minmaxmean[0],
                'max': minmaxmean[1],
                'mean': minmaxmean[2],
            })

        # for each property, compute its length
        maxlen = {}
        for p in props: maxlen[p] = 0
        for sp in str_props:
            for p in props:
                maxlen[p] = max(maxlen[p], len(sp[p]))

        # if any property got all empty strings, don't bother printing it, remove if from the list
        props = [p for p in props if maxlen[p] > 0]

        # print a header
        header_str = ""
        for p in props:
            prefix = "" if p == 'name' else " | "
            fmt_key = ">" if p == 'name' else "<"
            header_str += f"{prefix}{p:{fmt_key}{maxlen[p]}}"
        print(header_str)
        print("-" * len(header_str))

        # now print the acual arrays
        for strp in str_props:
            for p in props:
                prefix = "" if p == 'name' else " | "
                fmt_key = ">" if p == 'name' else "<"
                print(f"{prefix}{strp[p]:{fmt_key}{maxlen[p]}}", end='')
            print("")

    finally:
        del frame


def crop_to_extents_and_copy(src, dest, size=512, pad=False):
    img = Image.open(src)
    mask = np.array(img)[:, :, -1] > 0
    inds = np.where(mask)
    bounds_y = [inds[0].min(), inds[0].max()]
    bounds_x = [inds[1].min(), inds[1].max()]
    bounds_range_y = bounds_y[1] - bounds_y[0]
    bounds_range_x = bounds_x[1] - bounds_x[0]
    if bounds_range_y > bounds_range_x:
        top = bounds_y[0]
        bottom = bounds_y[1]
        left = (size - bounds_range_y) / 2
        right = (size + bounds_range_y) / 2
    else:
        left = bounds_x[0]
        right = bounds_x[1]
        top = (size - bounds_range_x) / 2
        bottom = (size + bounds_range_x) / 2
    img = img.crop((left, top, right, bottom))
    if pad:
        result = Image.new(img.mode, (int(size * 1.2), int(size * 1.2)), (255, 255, 255))
        result.paste(img, (int(size * 0.1), int(size * 0.1)))
        img = result
    img.resize((size, size), resample=Image.LANCZOS).save(dest)


def convert_to_video(path):
    path = Path(path)
    frames = sorted(list(path.iterdir()))
    output_path = path.parent / f"{path.stem}.mp4"
    collection = []
    for x in frames:
        img = np.array(Image.open(x))
        collection.append(img)
    clip = ImageSequenceClip(collection, fps=30)
    clip.write_videofile(str(output_path), verbose=False, logger=None)


if __name__ == "__main__":
    import sys
    convert_to_video(sys.argv[1])
