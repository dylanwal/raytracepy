"""
pip install numpy plotly, Pillow, kaleido==0.1.0post1

"""

import copy
import os
import pathlib

import numpy as np
import plotly.graph_objs as go
import PIL.Image
import PIL.ImageOps


def figs_to_grid_png(
        figs: list[go.Figure],
        shape: tuple[int, int],
        file_name: str = "grid.png",
        folder: str = "imgs",
        auto_open: bool = False,
        save_individual_imgs: bool = False
):
    """
    Converts Plotly figures into a single PNG by placing figures in grid.
    """
    make_new_folder(folder)

    # figs to png images
    imgs = []
    for i, fig in enumerate(figs):
        fig.write_image(folder + f"\\img{i}.png")  # pip install kaleido==0.1.0post1
        imgs.append(folder + f"\\img{i}.png")

    create_png_grid(imgs, shape, file_name, auto_open)

    if not save_individual_imgs:
        delete_files_and_folder(imgs)


def create_png_grid(imgs: list[str], shape: tuple[int, int], file_name: str = "molecule_grid.png",
                    auto_open: bool = True):
    imgs = copy.copy(imgs)

    img = PIL.Image.open(imgs[0])
    img_crop, crop_box = crop_off_white(img, padding=np.array([-10, -10, 10, 10], dtype="float64"),
                                        return_crop_box=True)
    cell_width = img_crop.width
    cell_height = img_crop.height

    new_im = PIL.Image.new('RGB', (cell_width * shape[0], cell_height * shape[1]))

    FLAG = True
    for row in range(shape[1]):
        for col in range(shape[0]):
            try:
                img = imgs.pop(0)
                img = PIL.Image.open(img).crop(crop_box)
            except IndexError:
                FLAG = False
                img = PIL.Image.new("RGB", (cell_width, cell_height), (255, 255, 255))

            new_im.paste(img, (col * cell_width, row * cell_height))

        if not FLAG:
            break



    new_im.save(file_name)

    if auto_open:
        new_im.show()


def crop_off_white(img: PIL.Image.Image, padding: np.ndarray = np.array([-10, -10, 10, 10], dtype="float64"),
                   return_crop_box: bool = False) -> PIL.Image.Image | tuple[PIL.Image.Image, np.ndarray]:
    """
    Removes white boarder around images.

    Parameters
    ----------
    img: PIL.Image
        image to be cropped
    padding: np.array
        padding left on image
    return_crop_box: bool
        return crop box

    Returns
    -------

    """
    # crop white
    invert_im = img.convert("RGB")  # remove alpha channel
    invert_im = PIL.ImageOps.invert(invert_im)  # invert image (so that white is 0)
    imageBox = invert_im.getbbox()
    imageBox = np.array(np.array(imageBox) + padding)

    # make sure crop doesn't go outside img limits
    imageBox[0] = np.max([0, imageBox[0]])
    imageBox[1] = np.max([0, imageBox[1]])
    imageBox[2] = np.min([img.width, imageBox[2]])
    imageBox[3] = np.min([img.height, imageBox[3]])

    if return_crop_box:
        return img.crop(imageBox), imageBox
    return img.crop(imageBox)


def make_new_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)


def delete_files_and_folder(files: list[str], remove_folder: bool = True):
    """ delete files and folder if empty. """
    folder = pathlib.Path(files[0]).parent
    # remove files
    for file in files:
        os.remove(file)
        # remove folder
    if remove_folder and len(os.listdir(folder)) == 0:
        os.rmdir(folder)
