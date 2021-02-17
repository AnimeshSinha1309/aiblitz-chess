import chess.engine
import ray

ray.shutdown()
ray.init()


def engine_init():
    return chess.engine.SimpleEngine.popen_uci("./stockfish")


def make_fen_valid(fen, move):
    fen += " "
    fen += move
    fen += " - - 0 2"
    return fen


VALID_MOVES = ["b", "w"]


def get_board(fen, move):
    try:
        return chess.Board(make_fen_valid(fen, move))
    except Exception as e:
        print("WRONG FEN", e)
        return chess.Board(chess.STARTING_BOARD_FEN)


@ray.remote
def evaluate_one(fen):
    engine = engine_init()
    engine.close()
    return 0


def others(fen):
    winner = None
    positions = [get_board(fen, move) for move in VALID_MOVES]

    for move, position in zip(VALID_MOVES, positions):
        if position.is_checkmate():
            winner = "w" if move == "b" else "b"
            break

    if winner is None:
        scores = []

        for move, position in zip(VALID_MOVES, positions):
            try:
                # Path to executable, get it from here https://stockfishchess.org/download/
                info = engine.analyse(position, chess.engine.Limit(time=1))

                # get a relative integer score from white's perspective, the score is in centi-pawns
                score = info["score"].white().score(mate_score=10000)

                # print("Score:", score, fen)
                scores.append(abs(score))
            except chess.engine.EngineTerminatedError:
                engine = engine_init()
                scores.append(0)
                # print(e)
                # print("This move is incorrect")

        winner = VALID_MOVES[0] if scores[0] > scores[1] else VALID_MOVES[1]

    assert (winner is not None)
    engine.quit()
    return winner


def evaluate(fen_list):
    fen_list = list(fen_list)[:10]

    results = []
    for fen, _turn in fen_list:
        results.append(evaluate_one.remote(fen))

    results = ray.get(results)

    return results
