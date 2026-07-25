#!/usr/bin/env python3
"""Convolution on images using multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    images: numpy.ndarray of shape (m, h, w, c)
    kernels: numpy.ndarray of shape (kh, kw, c, nc)
    padding: 'same', 'valid', or a tuple of (ph, pw)
    stride: tuple of (sh, sw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    convolved = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                v0 = i * sh
                v1 = v0 + kh
                h0 = j * sw
                h1 = h0 + kw
                image_slice = images_padded[:, v0:v1, h0:h1, :]
                kernel = kernels[:, :, :, k]
                convolved[:, i, j, k] = np.sum(
                    image_slice * kernel, axis=(1, 2, 3)
                )

    return convolved
