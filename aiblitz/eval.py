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


counter = 0

MATE_SCORE = 10000000


def score_move(engine, fen, move):
    position = get_board(fen, move)

    dbg = False
    # dbg = counter == 468

    if dbg:
        print(f"{fen} {move}")
        print(position.status())

    if position.is_checkmate():
        if dbg:
            print("mated")
        pos_score = -MATE_SCORE  # whoever is moving is losing
    elif (position.status() & chess.STATUS_OPPOSITE_CHECK) > 0:
        if dbg:
            print("mating others")
        pos_score = MATE_SCORE
    elif not position.is_valid():
        if dbg:
            print("invalid")
        pos_score = None  # nobody cares
    else:
        # changing engine time limit does not help improve accuracy
        info = engine.analyse(position, chess.engine.Limit(time=0.1))

        pos_score = info["score"]

        # get a relative integer score from mover's perspective, the score is in centi-pawns
        if move == 'w':
            pos_score = pos_score.white()
        else:
            pos_score = pos_score.black()

        pos_score = pos_score.score(mate_score=MATE_SCORE)
        # except chess.engine.EngineTerminatedError:
        #     print("Terminated", position, position.fen(), position.is_valid())
        #     assert False
        if dbg:
            print(f'Score: {pos_score}')

    return pos_score


def evaluate_one(engine, fen, turn):
    score0 = score_move(engine, fen, VALID_MOVES[0])
    score1 = score_move(engine, fen, VALID_MOVES[1])

    if score0 is None:
        winner = 1
    elif score1 is None:
        winner = 0
    else:
        use_score = score0 > score1 if turn == 0 else score1 > score0
        winner = turn if use_score > 0 else 1 - turn

    global counter
    counter += 1

    return winner


def evaluate(fen_list):
    fen_list = list(fen_list)

    engine = engine_init()
    results = [evaluate_one(engine, fen, turn) for fen, turn in tqdm.tqdm(fen_list)]
    engine.quit()

    return results
