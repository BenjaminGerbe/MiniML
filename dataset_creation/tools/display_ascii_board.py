
# Display part

import chess
import chess.pgn

# Stops the parsing after max_games
max_games = 5
incr = 0

pgn_path = "../test_files/lichess_db_standard_rated_2013-01.pgn"
with open(pgn_path) as pgn_file:
    for game in iter(lambda: chess.pgn.read_game(pgn_file), None):
        incr += 1
        print("Event:", game.headers["Event"])
        print("Site:", game.headers["Site"])
        print("Date:", game.headers["Date"])
        print("White:", game.headers["White"])
        print("Black:", game.headers["Black"])
        print("Result:", game.headers["Result"])

        board = game.board()
        for move in game.mainline_moves():
            board.push(move)

            # Print the board in ASCII format
            print("\n")
            print(board)
            print("\n")
        print("-" * 80)
        
        if incr == max_games:
            break

