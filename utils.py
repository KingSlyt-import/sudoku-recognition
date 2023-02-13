import cv2
import numpy as np
import pickle
import pandas as pd

# Pickle model for on a subset of the cell images 
# taken from the same sudoku
with open("pickle_model.pkl", "rb") as file:
    pickle_model = pickle.load(file)

def find_grid(img):
    """
    Given the image from input folder,
    scan for sudoku grid in an image
    """

    # Preprogress the image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 91, 3
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Calculate the contour length
        # Second parameter set to true cause Sudoku grid is closed
        perimeter = cv2.arcLength(contour, True)
        # Contour approximation
        approx = cv2.approxPolyDP(contour, 0.012 * perimeter, True)

        # If that contour have 4 edged points
        if (len(approx) == 4):
            warped, M = __four_point_transform(
                thresh,
                np.array([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
            )

            cells = find_cells(warped)
            # If the amount of cell in this grid is 81
            # We can safely said we found a sudoku grid
            if len(cells) == 81:
                return True, warped, M
        
    return False, warped, M

def __four_point_transform(img, pts):
    """
    Given the threshold version of the image and their 4 points of the contour,
    Correct it's perspective, 
    warp the image perspective into a top-down view
    """
    # Sorting 4 points
    rect = __order_points_of_quadrilateral(pts)
    (top_left, top_right, bottom_right, bottom_left) = rect

    # Compute the width of the new perspective
    widthA = np.sqrt(
        ((bottom_right[0] - bottom_left[0]) ** 2)
        + ((bottom_right[1] - bottom_left[1]) ** 2)
    )
    widthB = np.sqrt(
        ((top_right[0] - top_left[0]) ** 2) 
        + ((top_right[1] - top_left[1]) ** 2)
    )
    max_width = min(int(widthA), int(widthB))

    # Compute the height of the new perspective
    height_a = np.sqrt(
        ((top_right[0] - bottom_right[0]) ** 2)
        + ((top_right[1] - bottom_right[1]) ** 2)
    )
    height_b = np.sqrt(
        ((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2)
    )
    max_height = min(int(height_a), int(height_b))

    # Now we generate a set of destination points
    # to gain a top-down view of the original image
    dst = np.array(
        [
            [10, 10],
            [max_width - 10, 10],
            [max_width - 10, max_height - 10],
            [10, max_height - 10],
        ],
        dtype="float32",
    )

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (max_width + 10, max_height + 10))

    return warped, M

def __order_points_of_quadrilateral(pts):
    """
    Given 4 points of the contour,
    sorts them in the following order:
    (
        0: top-left, 
        1: top-right, 
        2: bottom-right, 
        3: bottom-left
    )
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Adding the x and y value of each point
    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Find the difference of the x and y value of each point
    # The top-right point will have the smallest difference
    # The bottom-left point will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def find_cells(img):
    """
    Given the threshold version of the image
    Find all the cells of a sudoku grid
    """
    img_area = img.shape[0] * img.shape[1]

    # Just like finding sudoku grid but in smaller scale
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Count all the cell
    cells = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Calculate the contour length
        # Second parameter set to true cause Sudoku grid is closed
        perimeter = cv2.arcLength(contour, True)
        # Contour approximation
        approx = cv2.approxPolyDP(contour, 0.017 * perimeter, True)

        # Filter for areas that are too small or too large in relation to the whole image
        if area / img_area > 0.0001 and area / img_area < 0.02 and len(approx) == 4:
            mask = np.zeros_like(img)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            (y, x) = np.where(mask == 255)

            (top_y, top_x) = (np.min(y), np.min(x))
            (bottom_y, bottom_x) = (np.max(y), np.max(x))
            cell = img[top_y : bottom_y + 1, top_x : bottom_x + 1]

            cell = cell.copy()
            cell = cv2.resize(cell, (28, 28))

            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cells.append(({"img": cell, "pos": (cX, cY)}))

    return cells

def find_cell_values(cells):
    """
    Given the image of each cell (28x28)px
    determine it's value according to the pickle model
    """
    cells_with_value = []
    for cell in cells:
        y_pred = pickle_model.predict([cell["img"].flatten()])[0]
        cells_with_value.append({"pos": cell["pos"], "value": y_pred})

    return cells_with_value

def get_sudoku_grid(cells):
    """
    Given a list of cells and they position, return a 2D array representing
    a sudoku grid, where each element of this 2D array contains of the value of the grid
    at that position.
    """
    cells.sort(key=lambda cell: cell["pos"][0])

    cells_X = []
    i = 0
    for cell in cells:
        cells_X.append(
            {"pos": cell["pos"], "value": cell["value"], "columnIndex": int(i / 9)}
        )
        i += 1

    cells_X.sort(key=lambda cell: cell["pos"][1])

    cells_in_grid = []
    i = 0
    for cell in cells_X:
        cells_in_grid.append(
            {
                "pos_in_grid": (cell["columnIndex"], int(i / 9)),
                "value": cell["value"],
                "pos": cell["pos"],
            }
        )
        i += 1

    grid = np.zeros((9, 9))
    grid_meta = np.zeros((9, 9, 3))
    for cell in cells_in_grid:
        grid[cell["pos_in_grid"][1], cell["pos_in_grid"][0]] = cell["value"]
        # We keep some info about the grid: centroid of each cell, and whether it is blank
        grid_meta[cell["pos_in_grid"][1], cell["pos_in_grid"][0]] = [
            cell["pos"][0],
            cell["pos"][1],
            cell["value"] == 0,
        ]

    return grid