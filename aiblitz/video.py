import cv2

import torch

from aiblitz.model import Net
from aiblitz.predictor import predict_batch
from aiblitz.segment import store_fen

import chess

"""
Frames are read from mp4 file present at dir_in/filename.mp4
and the frames are extracted to dir_out/filename/frameN.jpg
The first frame is recorded at 0ms, and every next frame is
with a delay of DELAYms
"""


def extract_images(file_id, model, delay=100):
    # Get the predictions
    path = f"data/Q4/train/{file_id}.mp4"
    video_capture = cv2.VideoCapture(path)
    frames = []
    for index in range(100):
        video_capture.set(cv2.CAP_PROP_POS_MSEC, (index * delay))
        success, image_frame = video_capture.read()
        if not success:
            break
        frames.append(image_frame)
    results = predict_batch(model, frames)
    # Convert to FEN
    fen_list = []
    for result in results:
        fen = store_fen(result)
        if len(fen_list) < 1 or fen != fen_list[-1]:
            fen_list.append(fen)
    return fen_list


def predict_move(file_id, model):
    fen_list = extract_images(file_id, model)
    # TODO: Convert FEN-List to move-list
    print(fen_list)  # Dummy statement, write code here
    move_list = []
    return move_list


"""
turn must be "white" or "black"
"""


def give_move(from_fen, to_fen, turn):
    from_fen += " " + turn[0]
    pos1 = chess.Board(from_fen)
    for move in pos1.legal_moves:
        pos2 = pos1.copy()
        pos2.push(move)
        if pos2.board_fen() == to_fen:
            move = pos2.pop()
            return move.uci()
    assert False


def give_moves(fen_list, initial_turn):
    prev_fen = fen_list[0]
    moves = []
    turn = initial_turn
    for i, fen in enumerate(fen_list[1:]):
        moves.append(give_move(prev_fen, fen, turn))
        prev_fen = fen
        turn = "black" if turn == "white" else "white"

    return moves


def tests_move():
    from_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    to_fen = "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR"
    assert give_move(from_fen, to_fen, "white") == "b1c3"
    from_fen = "rnbqkbnr/pp1p1ppp/8/2p1p3/5P2/2NP4/PPP1P1PP/R1BQKBNR"
    to_fen = "rnb1kbnr/pp1p1ppp/8/2p1p3/5P1q/2NP4/PPP1P1PP/R1BQKBNR"
    assert give_move(from_fen, to_fen, "black") == "d8h4"
    print("All tests passed")


if __name__ == "__main__":
    network = Net()
    network.load_state_dict(torch.load("weights/piece-recognizer.h5"))
    print(predict_move(1, network))
    # extract_images(1)
    tests_move()
