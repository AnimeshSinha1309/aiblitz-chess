import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv


def segment_image(path, result_size=32):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (8 * result_size, 8 * result_size))
    image = image.reshape((8, result_size, 8, result_size)).transpose(0, 2, 1, 3)
    return image


def show_grid(grid):
    plt.figure(figsize=(16, 16))
    for i in range(8):
        for j in range(8):
            plt.subplot(8, 8, i * 8 + j + 1)
            plt.imshow(grid[i, j], cmap='gray')
    plt.show()


piece_to_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
                'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11, ' ': 12}
idx_to_piece = {value: key for key, value in piece_to_idx.items()}


def parse_fen(fen):
    result = np.full(shape=(8, 8), fill_value=piece_to_idx[' '])
    for i, row in enumerate(fen.split('/')):
        j = 0
        for char in row:
            if char in "12345678":
                j += int(char)
            else:
                result[i, j] = piece_to_idx[char]
                j += 1
    return result


def store_fen(board):
    result = []
    for row in board:
        row_val, gap = "", 0
        for cell in row:
            piece = idx_to_piece[cell]
            if piece == ' ':
                gap += 1
            else:
                if gap > 0:
                    row_val += str(gap)
                    gap = 0
                row_val += piece
        if gap > 0:
            row_val += str(gap)
        result.append(row_val)
    return "/".join(result)


if __name__ == "__main__":
    df_train = pd.read_csv("../data/Q3/train.csv")
    df_val = pd.read_csv("../data/Q3/val.csv")
    x = parse_fen(df_train["label"][0])

    image_path = "../data/Q3/train/%d.jpg" % df_train["ImageID"][0]
    segmented_image = segment_image(image_path)
    show_grid(segmented_image)
    print(x)
