import chess
import chess.pgn

max_games = 2
incr = 0

pgn_path = "../test_files/test_evals_500.pgn"
pgn_file = open(pgn_path)

for game in iter(lambda: chess.pgn.read_game(pgn_file), None):
    incr += 1
    print("Event:", game.headers["Event"])
    print("Site:", game.headers["Site"])
    print("Date:", game.headers["Date"])
    print("White:", game.headers["White"])
    print("Black:", game.headers["Black"])
    print("Result:", game.headers["Result"])

    board = game.board()
    root = game.root()

    for move in game.mainline_moves():
        # Picking up the score of the current node and printing it
        child = root.variation(0)
        evaluation = child.eval()
        print(move, evaluation)
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

        # Inversion of the board to match the ascii output
        for row in board_state:
            print(row [::-1])

        print("\n")
        print("-" * 80)
        print("\n")
        
        if incr == max_games:
            break

    if incr == max_games:
        break
