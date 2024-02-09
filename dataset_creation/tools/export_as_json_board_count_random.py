import chess
import chess.pgn
import json
import random

max_board_states = 50000
board_state_count = 0

pgn_path = "../test_files/10k_analysed_games.pgn"
pgn_file = open(pgn_path)

games = []

all_games = list(iter(lambda: chess.pgn.read_game(pgn_file), None))

random.shuffle(all_games)

for game in all_games:
    game_info = {}
    board = game.board()
    root = game.root()
    turns = []
    #print("Site:", game.headers["Site"])

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

        find = False
      
         
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
        board_state = [int(bit) for row in board_state for cell in row for bit in cell]

        for t in games:
            if(t['evaluation'] == turn_info["evaluation"] and t['move'] == turn_info["move"]):
                find = True;
                break;
            
        if(not find):
            turn_info["board_state"] = board_state
            games.append(turn_info)
            board_state_count += 1
        
        if board_state_count == max_board_states:
            break

    #game_info["turns"] = turns
    #games.append(game_info)
    if board_state_count == max_board_states:
        break

with open("../output_files/games.json", "w") as f:
    json_string = json.dumps(games)
    json_string = json_string.replace('"}','"}')
    f.write(json_string)
    print("finish")
