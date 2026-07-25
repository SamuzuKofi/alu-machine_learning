#!/usr/bin/env python3
"""Strided convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)
    padding: 'same', 'valid', or a tuple of (ph, pw)
    stride: tuple of (sh, sw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        out_h = -(-h // sh)
        out_w = -(-w // sw)
        ph = max(((out_h - 1) * sh + kh - h) // 2, 0)
        pw = max(((out_w - 1) * sw + kw - w) // 2, 0)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    convolved = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            v_start = i * sh
            h_start = j * sw
            v_end = v_start + kh
            h_end = h_start + kw
            image_slice = images_padded[:, v_start:v_end, h_start:h_end]
            convolved[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convolved
