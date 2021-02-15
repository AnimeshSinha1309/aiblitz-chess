import chess.engine


def evaluate(fen_list):
    results = []
    for fen, turn in fen_list:
        engine = chess.engine.SimpleEngine.popen_uci("./stockfish")
        position = chess.Board(fen)
        # Path to executable, get it from here https://stockfishchess.org/download/
        info = engine.analyse(position, chess.engine.Limit(time=0.1))
        # get a relative integer score from white's perspective, the score is in centi-pawns
        score = info["score"].white().score(mate_score=10000)
        print("Score:", score, fen)
        # As per AICrowd, one of the sides is certainly winning/mating
        # If this assertion fails, then the engine had insufficient time to analyze the position
        assert(abs(score) >= 150)
        # Print winner side
        results.append("white" if score > 0 else "black")

    return results
