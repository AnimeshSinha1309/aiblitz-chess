import chess.engine
import tqdm

WHITE = 0
BLACK = 1
VALID_MOVES = ["w", "b"]


def engine_init():
    # Path to executable, get it from here https://stockfishchess.org/download/
    return chess.engine.SimpleEngine.popen_uci("./stockfish")


def make_fen_valid(fen, move):
    fen += " "
    fen += move
    fen += " - - 0 2"
    return fen


def get_board(fen, move):
    try:
        return chess.Board(make_fen_valid(fen, move))
    except Exception as e:
        print("WRONG FEN", e)
        return chess.Board(chess.STARTING_BOARD_FEN)


def score_move(engine, fen, move):
    position = get_board(fen, move)

    if position.is_checkmate():
        pos_score = -10000000000  # whoever is moving is losing
    elif not position.is_valid():
        pos_score = 0  # nobody cares
    else:
        info = engine.analyse(position, chess.engine.Limit(time=0.1))

        pos_score = info["score"]

        # get a relative integer score from mover's perspective, the score is in centi-pawns
        if move == 'w':
            pos_score = pos_score.white()
        else:
            pos_score = pos_score.black()

        pos_score = pos_score.score(mate_score=10000)
        # except chess.engine.EngineTerminatedError:
        #     print("Terminated", position, position.fen(), position.is_valid())
        #     assert False

    assert (pos_score is not None)
    return pos_score


def evaluate_one(engine, fen):
    score0 = score_move(engine, fen, VALID_MOVES[0])
    score1 = score_move(engine, fen, VALID_MOVES[1])

    return 0 if score0 > score1 else 1


def evaluate(fen_list):
    fen_list = list(fen_list)

    engine = engine_init()
    results = [evaluate_one(engine, fen) for fen, _turn in tqdm.tqdm(fen_list)]
    engine.quit()

    return results
