import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import os
from sys import argv, stderr


MAX_VAL = 255
GRAYSCALE_REPRESENTATION = 1


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def read_image(filename, representation):
    """
    This function read an image from a given file path in a given representation
    :param filename: the file path to read the image from. Can be grayscale or RGB
    :param representation: define the output image: 1 - grayscale, 2 - RGB
    :return: the image in the given representation
    """
    image = imread(filename)
    if representation == GRAYSCALE_REPRESENTATION and len(image.shape) == 3:
        image = rgb2gray(image)
    if image.max() > 1:
        image = image / MAX_VAL
    return image.astype(np.float64)


def expand_image(image, filter_vec):
    """
    This function expands a given image by 2, after padding it with zeros, and then smooths it with a given
     gaussian filter
    :param image: the image to expand
    :param filter_vec: the gaussian filter
    :return: the expanded image
    """
    pad = np.zeros(2 * np.array(image.shape), dtype=image.dtype)
    pad[::2, ::2] = image
    normalized_filter = 2 * filter_vec
    return convolve(convolve(pad, normalized_filter), normalized_filter.T)


def reduce_image(image, filter_vec):
    """
    This function reduces a given image by half, after smoothing it with a given gaussian filter
    :param image: the image to reduce
    :param filter_vec: the gaussian filter
    :return: the reduced image
    """
    blur_im = convolve(convolve(image, filter_vec), filter_vec.T)
    return blur_im[::2, ::2]


def get_gaussian_filter(size):
    """
    This function generates a gaussian filter of a given size
    :param size: the size of the filter
    :return: the filter
    """
    if size == 1:
        return np.array([[1]])
    conv_vec = np.array([1, 1])
    res = conv_vec
    for i in range(size - 2):
        res = np.convolve(res, conv_vec)
    return np.array([res / sum(res)])


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function builds a gaussian pyramid from a given image
    :param im: a grayscale image with double values in [0; 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be
           used in constructing the pyramid filter
    :return: [pyramid (list of layers), filter vector (the filter that was used in the pyramid construction)]
    """
    pyr = [im]
    filter_vec = get_gaussian_filter(filter_size)
    for i in range(max_levels - 1):
        new_im = reduce_image(pyr[i], filter_vec)
        pyr.append(new_im)
        if new_im.shape[0] <= 32 or new_im.shape[1] <= 32:
            break
    return [pyr, filter_vec]


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    This function builds a laplacian pyramid from a given image
    :param im: a grayscale image with double values in [0; 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be
           used in constructing the pyramid filter
    :return: [pyramid (list of layers), filter vector (the filter that was used in the pyramid construction)]
    """
    gauss_pyr, gauss_filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    lap_pyr = []
    for lvl in range(len(gauss_pyr) - 1):
        exp_g = expand_image(gauss_pyr[lvl + 1], gauss_filter_vec)
        lap_pyr.append(gauss_pyr[lvl] - exp_g)
    lap_pyr.append(gauss_pyr[-1])
    return [lap_pyr, gauss_filter_vec]


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    This function performs a reconstruction of an image from its Laplacian Pyramid.
    :param lpyr: the Laplacian pyramid
    :param filter_vec: the filter that was used to create the pyramid
    :param coeff: a python list, with length the same as the number of levels in the pyramid lpyr.
    :return: the reconstructed image
    """
    res_im = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr) - 2, -1, -1):
        res_im = (lpyr[i] * coeff[i]) + expand_image(res_im, filter_vec)
    return res_im


def render_pyramid(pyr, levels):
    """
    This function creates a single black image in which the pyramid levels of the given pyramid pyr are
    stacked horizontally (after stretching the values to [0; 1])
    :param pyr: a Gaussian or Laplacian pyramid
    :param levels: the number of levels to present in the result (<= max_levels)
    :return: the created image
    """
    base_im_height = pyr[0].shape[0]
    res = np.interp(pyr[0], (pyr[0].min(), pyr[0].max()), (0.0, 1.0))
    for lvl in range(1, min(levels, len(pyr))):
        stretched = np.interp(pyr[lvl], (pyr[lvl].min(), pyr[lvl].max()), (0.0, 1.0))
        pad_height = base_im_height - pyr[lvl].shape[0]
        padded_im = np.pad(stretched, ((0, pad_height), (0, 0)), 'constant', constant_values=0)
        res = np.hstack((res, padded_im))
    return res


def display_pyramid(pyr, levels):
    """
    This function displays the given pyramid
    :param pyr: a Gaussian or Laplacian pyramid
    :param levels: the number of levels to present in the result (<= max_levels)
    """
    pyr_im = render_pyramid(pyr, levels)
    plt.imshow(pyr_im, cmap='gray')


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    This function blends two grayscale images into one using a mask and pyramids
    :param im1: the first grayscale image to be blended.
    :param im2: the second grayscale image to be blended.
    :param mask: a boolean (np.bool) mask containing True and False representing which parts of im1 and im2
           should appear in the result
    :param max_levels: the parameter to use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: the size of the Gaussian filter to use in the construction of the Laplacian
           pyramids of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter to use in the construction of the Gaussian
           pyramid of mask.
    :return: the blended image
    """
    l1, filter_im = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, filter_im = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gm, filter_mask = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = []
    for lvl in range(len(l1)):
        out_level = gm[lvl] * l1[lvl] + (1 - gm[lvl]) * l2[lvl]
        l_out.append(out_level)
    coeff = [1] * len(l_out)
    blended_im = laplacian_to_image(l_out, filter_im, coeff)
    return np.clip(blended_im, 0.0, 1.0)


def blend_rgb(im1, im2, mask, max_levels=3, filter_size_im=3, filter_size_mask=3):
    """
    This function blends two rgb images into one using a mask and pyramids
    :param im1: the first grayscale image to be blended.
    :param im2: the second grayscale image to be blended.
    :param mask: a boolean (np.bool) mask containing True and False representing which parts of im1 and im2
           should appear in the result
    :param max_levels: the parameter to use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: the size of the Gaussian filter to use in the construction of the Laplacian
           pyramids of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter to use in the construction of the Gaussian
           pyramid of mask.
    :return: the blended image
    """
    red = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size_im, filter_size_mask)
    green = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_levels, filter_size_im, filter_size_mask)
    blue = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_levels, filter_size_im, filter_size_mask)
    result = np.stack([red, green, blue], axis=-1)
    return result


def blending_example1():
    img1 = read_image(relpath("externals/cookie_monster.jpg"), 2)
    img2 = read_image(relpath("externals/wonder_woman.jpg"), 2)
    mask = read_image(relpath("externals/wonder_woman_mask.jpg"), 1)
    mask = np.round(mask)
    mask = mask.astype(np.bool)
    blended = blend_rgb(img1, img2, mask, 3, 3, 3)
    fig, a = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Cookie Monster Terror')
    a[0][0].imshow(img1, cmap='gray')
    a[0][1].imshow(img2, cmap='gray')
    a[1][0].imshow(mask, cmap='gray')
    a[1][1].imshow(blended, cmap='gray')
    return [img1, img2, mask, blended]


def blending_example2():
    img1 = read_image(relpath("externals/hermione.jpg"), 2)
    img2 = read_image(relpath("externals/statue.jpg"), 2)
    mask = read_image(relpath("externals/hermione_mask.jpg"), 1)
    mask = np.round(mask)
    mask = mask.astype(np.bool)
    blended = blend_rgb(img1, img2, mask, 3, 3, 5)
    fig, a = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Hermione of Liberty')
    a[0][0].imshow(img1, cmap='gray')
    a[0][1].imshow(img2, cmap='gray')
    a[1][0].imshow(mask, cmap='gray')
    a[1][1].imshow(blended, cmap='gray')
    return [img1, img2, mask, blended]


def blender():
    """
    The main function. Gets the required images and mask from the command line, and produce a plot with the
    results.
    """
    img1 = read_image(relpath(argv[1]), 2)
    img2 = read_image(relpath(argv[2]), 2)
    mask = read_image(relpath(argv[3]), 1)
    mask = np.round(mask)
    mask = mask.astype(np.bool)
    if len(argv) == 7:
        result = blend_rgb(img1, img2, mask, argv[4], argv[5], argv[6])

    else:
        result = blend_rgb(img1, img2, mask)
    plt.imshow(result, cmap='gray')


if __name__ == '__main__':
    if len(argv) == 2:
        if argv[1] == "example1":
            blending_example1()
            plt.show()
        elif argv[1] == "example2":
            blending_example2()
            plt.show()
    elif len(argv) == 4 or len(argv) == 7:
        for arg in argv[1:4]:
            filename, extension = os.path.splitext(arg)
            if extension != '.jpg':
                print(f"USAGE: The file {arg} should be of type '.jpg'", file=stderr)
                exit(1)
        if len(argv) == 7:
            if not argv[4].isdigit() or int(argv[4]) > 11:
                print(f"USAGE: {argv[4]} should be an integer in the range [1, 11]", file=stderr)
                exit(1)
            for arg in argv[5:]:
                if not arg.isdigit() or int(arg) % 2 != 1 or int(arg) > 13:
                    print(f"USAGE: {arg} should be an odd integer in the range [1, 13]", file=stderr)
                    exit(1)
        blender()
        plt.show()
    else:
        print(f"USAGE: you didn't enter the arguments correctly. Check out the README for the instructions")
        exit(1)
