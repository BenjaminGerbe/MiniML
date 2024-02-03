import chess
import chess.pgn
import json

max_games = 20
incr = 0

pgn_path = "../test_files/test_evals_500.pgn"
pgn_file = open(pgn_path)

games = []

for game in iter(lambda: chess.pgn.read_game(pgn_file), None):
    incr += 1
    game_info = {}
    board = game.board()
    root = game.root()
    turns = []

    for move in game.mainline_moves():
        if board.is_game_over():
            break
        turn_info = {}
        child = root.variation(0)
        if child.eval() is None:
            break
        evaluation = child.eval().relative
        turn_info["move"] = move.uci()
        turn_info["evaluation"] = str(evaluation)
        
        root = child
        board.push(move)
        board_state = []
        for row in range(8):
            row_state = []
            for col in range(8):
                cell_state = ""
                for piece_type in chess.PIECE_TYPES:
                    for color in chess.COLORS:
                        piece = board.pieces(piece_type, color)
                        piece_int = int(piece)
                        # Convertion to a binary string
                        piece_bin = bin(piece_int)
                        # Removing the '0b' prefix
                        piece_bin = piece_bin[2:]
                        # Pad with zeros to make it 64 digits long
                        piece_bin = piece_bin.zfill(64)
                        # Get the bit corresponding to the current cell
                        bit = piece_bin[8 * row + col]
                        # Append the bit to the cell state
                        cell_state += bit
                row_state.append(cell_state)
            board_state.append(row_state)

        board_state = [row[::-1] for row in board_state]
        
        turn_info["board_state"] = board_state
        turns.append(turn_info)
        
        if incr == max_games:
            break

    game_info["turns"] = turns
    games.append(game_info)

    if incr == max_games:
        break

with open("../output_files/games.json", "w") as f:
    json.dump(games, f, indent=4)
