import cv2

import torch

from aiblitz.models.conv import Net
from aiblitz.segment import store_fen, segment_image

import chess


def predict_batch(model, image_frames):
    with torch.no_grad():
        model.eval()
        images = [torch.from_numpy(segment_image(image_frame)) for image_frame in image_frames]
        x = torch.stack(images)
        x = x.view(-1, 3, 32, 32).float()
        y = model(x)
        y = torch.argmax(y, -1)
        y = y.view(-1, 8, 8)
    return y.numpy()


def extract_images(file_id, model, folder="test", delay=100):
    # Get the predictions
    path = f"data/Q4/{folder}/{file_id}.mp4"
    video_capture = cv2.VideoCapture(path)
    frames = []
    for index in range(100):
        video_capture.set(cv2.CAP_PROP_POS_MSEC, (index * delay))
        success, image_frame = video_capture.read()
        if not success:
            break
        frames.append(image_frame)
    if len(frames) == 0:
        return []
    results = predict_batch(model, frames)
    # Convert to FEN
    fen_list = []
    for result in results:
        fen = store_fen(result)
        if len(fen_list) < 1 or fen != fen_list[-1]:
            fen_list.append(fen)
    return fen_list


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


def predict_move(file_id, model):
    fen_list = extract_images(file_id, model)
    if len(fen_list) <= 1:
        print(f"{len(fen_list)} frames extracted from video file {file_id}")
        return []
    # TODO: Convert FEN-List to move-list
    for initial_turn in ["black", "white"]:
        prev_fen = fen_list[0]
        moves = []
        successful_parsing = True
        turn = initial_turn
        for i, fen in enumerate(fen_list[1:]):
            try:
                moves.append(give_move(prev_fen, fen, turn))
            except AssertionError:
                successful_parsing = False
                break
            prev_fen = fen
            turn = "black" if turn == "white" else "white"
        if successful_parsing:
            return moves

    print("Could not Parse video file", file_id)
    return []


if __name__ == "__main__":
    network = Net()
    network.load_state_dict(torch.load("weights/piece-recognizer.h5"))
    print(predict_move(0, network))
