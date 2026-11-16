import numpy as np
import skimage.transform


def process_chunk(args):
    idata, odata, z, y, x, cz, cy, cx, algorithm = args
    starts = (2*z*cz, 2*y*cy, 2*x*cx)
    stops = tuple(min(start + 2*chunk, size) for start, chunk, size in zip(starts, (cz, cy, cx), idata.shape))
    halo = 2 if algorithm == 'gaussian' else 0
    read_starts = tuple(max(0, start - halo) for start in starts)
    read_stops = tuple(min(size, stop + halo) for stop, size in zip(stops, idata.shape))
    ibuf = idata[tuple(slice(start, stop) for start, stop in zip(read_starts, read_stops))]
    if np.max(ibuf) == 0:
        return
    pad = tuple(size % 2 for size in ibuf.shape)
    if any(pad):
        ibuf = np.pad(ibuf, tuple((0, amount) for amount in pad), mode='symmetric')
    if algorithm == 'nearest':
        obuf = ibuf[::2, ::2, ::2]
    elif algorithm == 'gaussian':
        obuf = np.round(skimage.transform.rescale(ibuf, .5, preserve_range=True))
        offsets = tuple((start - read_start) // 2 for start, read_start in zip(starts, read_starts))
        lengths = tuple((stop - start + 1) // 2 for start, stop in zip(starts, stops))
        obuf = obuf[tuple(slice(offset, offset + length) for offset, length in zip(offsets, lengths))]
    elif algorithm == 'mean':
        obuf = np.round(skimage.transform.downscale_local_mean(ibuf, (2, 2, 2)))
    elif algorithm == 'max':
        s = ibuf.shape
        obuf = np.max(ibuf.reshape((s[0]//2, 2, s[1]//2, 2, s[2]//2, 2)), axis=(1, 3, 5))
    else:
        raise ValueError(f'algorithm {algorithm} not valid')
    odata[z*cz:(z+1)*cz, y*cy:(y+1)*cy, x*cx:(x+1)*cx] = np.round(obuf)
