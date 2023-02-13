import cv2
import glob
import os

import utils

def sudoku_processing(file_path):
    # Image processing in folder
    img = cv2.imread(file_path)

    # Image ratio for consistent processing size
    img_ratio = img.shape[0] / img.shape[1]
    img = cv2.resize(img, (1100, int(1100 * img_ratio)))

    # Scan the image sudoku grid in image
    valid, img_grid, M = utils.find_grid(img) 
    # If the function cannot find any grid on image
    # We move on to the next one
    if not valid:
        return False

    # Generate a 2D array representation of the grid present in the image
    cells = utils.find_cells(img_grid)
    cells_with_value = utils.find_cell_values(cells)
    grid, grid_meta = utils.get_sudoku_grid(cells_with_value)

    print(grid)