import pygame
import sys
import json
import os
try:
    import pyperclip
except ImportError:
    print("Installing pyperclip...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyperclip"])
    import pyperclip

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BOARD_SIZE = 512  # Size of the chess board (8x8 squares)
SQUARE_SIZE = BOARD_SIZE // 8
UI_AREA_HEIGHT = 100  # Height of the UI area below the board
BUTTON_HEIGHT = 30
INPUT_BOX_HEIGHT = 30
MESSAGE_DURATION = 3000  # Duration to show messages in milliseconds (3 seconds)
PIECE_SIZE = 65  # Size for piece images
PROMOTION_RECT_SIZE = 50  # Size of promotion selection squares
MIN_WINDOW_WIDTH = 400
MIN_WINDOW_HEIGHT = 480

# Colors
BOARD_COLOR_LIGHT = (240, 217, 181)   # Light squares
BOARD_COLOR_DARK = (181, 136, 99)     # Dark squares
TEXT_COLOR = (0, 0, 0)                # Black text
BUTTON_COLOR = (200, 200, 200)        # Default button color
BUTTON_HOVER_COLOR = (180, 180, 180)  # Button color when hovered
HIGHLIGHT_COLOR = (100, 200, 100, 128) # Semi-transparent green for move highlights
MESSAGE_COLOR_SUCCESS = (0, 150, 0)    # Green color for success messages
MESSAGE_COLOR_ERROR = (150, 0, 0)      # Red color for error messages
INPUT_BOX_COLOR = (255, 255, 255)
INPUT_BOX_ACTIVE_COLOR = (240, 240, 255)

# Game constants
SAVE_FILE = "chess_save.json"
PROMOTION_PIECES = ['q', 'r', 'n', 'b']  # Available promotion pieces

# Initial window dimensions
width = WINDOW_WIDTH
height = WINDOW_HEIGHT

# Set up the display
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption("Chess Interface")

# Load piece images
piece_images = {}
try:
    print("Starting to load piece images...")  # Debug message
    # Mapping of piece letters to image file names
    piece_files = {
        'K': 'white_king.png',
        'Q': 'white_queen.png',
        'B': 'white_bishop.png',
        'N': 'white_knight.png',
        'R': 'white_rook.png',
        'P': 'white_pawn.png',
        'k': 'black_king.png',
        'q': 'black_queen.png',
        'b': 'black_bishop.png',
        'n': 'black_knight.png',
        'r': 'black_rook.png',
        'p': 'black_pawn.png'
    }
    
    # Get the absolute path to the assets directory
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, 'assets')
    print(f"Looking for assets in: {assets_dir}")  # Debug message
    
    if not os.path.exists(assets_dir):
        print(f"Assets directory not found at: {assets_dir}")
    else:
        for piece, filename in piece_files.items():
            file_path = os.path.join(assets_dir, filename)
            if os.path.exists(file_path):
                try:
                    print(f"Loading: {filename}")  # Debug message
                    # Load and scale each piece image
                    image = pygame.image.load(file_path)
                    piece_images[piece] = pygame.transform.scale(image, (PIECE_SIZE, PIECE_SIZE))
                    print(f"Successfully loaded: {filename}")  # Debug message
                except pygame.error as e:
                    print(f"Error loading {filename}: {e}")
                except Exception as e:
                    print(f"Unexpected error loading {filename}: {e}")
            else:
                print(f"Missing image file: {filename}")
except Exception as e:
    print(f"Error in image loading setup: {e}")

print(f"Loaded {len(piece_images)} piece images")

# Initialize fonts
title_font = pygame.font.Font(None, 36)  # Larger font for titles
text_font = pygame.font.Font(None, 24)   # Regular text font
turn_font = pygame.font.Font(None, 28)   # Medium font for turn indicator
fallback_font = pygame.font.Font(None, 48)  # Create a font for displaying the placeholders

# Piece placeholders (fallback for missing images)
piece_placeholders = {
    "r": "r", "n": "n", "b": "b", "q": "q", "k": "k", "p": "p",
    "R": "R", "N": "N", "B": "B", "Q": "Q", "K": "K", "P": "P"
}

# Initial board setup
board = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"],
]

# Game state variables
selected_piece = None
selected_piece_pos = None
turn = "white"
castling_rights = "KQkq"
en_passant_target = "-"
halfmove_clock = 0
fullmove_number = 1
white_king_pos = (7, 4)
black_king_pos = (0, 4)

# UI state variables
show_moves_popup = False
input_box_active = False
button_hovered = False
fen_text = ""
message_text = ""
message_color = TEXT_COLOR
message_start_time = 0
paste_button_rect = None
copy_button_rect = None
input_box_rect = None
undo_button_rect = None
redo_button_rect = None
possible_moves = []

# Promotion state variables
promotion_squares = None
awaiting_promotion = False
promotion_start = None
promotion_end = None

# Move history for undo/redo
move_history = []
current_move_index = -1
algebraic_moves = []

def init_pygame():
    """Initialize pygame and fonts"""
    global title_font, text_font, turn_font
    pygame.init()
    title_font = pygame.font.Font(None, 36)  # Larger font for titles
    text_font = pygame.font.Font(None, 24)   # Regular text font
    turn_font = pygame.font.Font(None, 28)   # Medium font for turn indicator

def copy_moves_to_clipboard():
    """Copy the move history to clipboard"""
    try:
        moves_text = ""
        for i in range(0, len(algebraic_moves), 2):
            move_number = i // 2 + 1
            white_move = algebraic_moves[i]
            black_move = algebraic_moves[i + 1] if i + 1 < len(algebraic_moves) else ""
            moves_text += f"{move_number}. {white_move} {black_move}\n"
        pyperclip.copy(moves_text.strip())
        message_text = "Moves copied to clipboard"
        message_color = MESSAGE_COLOR_SUCCESS
        message_start_time = pygame.time.get_ticks()
    except Exception as e:
        message_text = "Error copying moves to clipboard"
        message_color = MESSAGE_COLOR_ERROR
        message_start_time = pygame.time.get_ticks()

def save_game_state():
    """Save current game state for undo/redo"""
    global current_move_index, move_history
    # Create deep copies of the current state
    board_copy = [row[:] for row in board]
    state = {
        'board': board_copy,
        'turn': turn,
        'castling_rights': castling_rights,
        'en_passant_target': en_passant_target,
        'halfmove_clock': halfmove_clock,
        'fullmove_number': fullmove_number,
        'white_king_pos': white_king_pos,
        'black_king_pos': black_king_pos
    }
    
    # If we're not at the end of the history, truncate the future moves
    if current_move_index < len(move_history) - 1:
        move_history = move_history[:current_move_index + 1]
        algebraic_moves = algebraic_moves[:current_move_index + 1]
    
    move_history.append(state)
    current_move_index = len(move_history) - 1

def undo_move():
    """Undo the last move"""
    global board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number
    global white_king_pos, black_king_pos, current_move_index
    
    if current_move_index > 0:
        current_move_index -= 1
        state = move_history[current_move_index]
        board = [row[:] for row in state['board']]
        turn = state['turn']
        castling_rights = state['castling_rights']
        en_passant_target = state['en_passant_target']
        halfmove_clock = state['halfmove_clock']
        fullmove_number = state['fullmove_number']
        white_king_pos = state['white_king_pos']
        black_king_pos = state['black_king_pos']
        return True
    return False

def redo_move():
    """Redo the previously undone move"""
    global board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number
    global white_king_pos, black_king_pos, current_move_index
    
    if current_move_index < len(move_history) - 1:
        current_move_index += 1
        state = move_history[current_move_index]
        board = [row[:] for row in state['board']]
        turn = state['turn']
        castling_rights = state['castling_rights']
        en_passant_target = state['en_passant_target']
        halfmove_clock = state['halfmove_clock']
        fullmove_number = state['fullmove_number']
        white_king_pos = state['white_king_pos']
        black_king_pos = state['black_king_pos']
        return True
    return False

def calculate_possible_moves(board, row, col, turn, white_king_pos, black_king_pos, castling_rights, en_passant_target):
    """Calculate all possible moves for a piece"""
    piece = board[row][col]
    possible_moves = []
    
    # Check if it's the right turn
    if piece == ' ' or (turn == 'white' and piece.islower()) or (turn == 'black' and piece.isupper()):
        return possible_moves
    
    # Calculate all potential moves first
    potential_moves = []
    
    # Pawn moves
    if piece.lower() == 'p':
        direction = 1 if piece == 'p' else -1  # Black pawns move down, white pawns move up
        start_row = 1 if piece == 'p' else 6   # Starting row for pawns
        
        # Normal move forward
        next_row = row + direction
        if 0 <= next_row < 8 and board[next_row][col] == ' ':
            potential_moves.append((next_row, col))
            # Initial two-square move
            if row == start_row and board[row + 2*direction][col] == ' ':
                potential_moves.append((row + 2*direction, col))
        
        # Diagonal captures
        for dcol in [-1, 1]:
            new_col = col + dcol
            new_row = row + direction
            if 0 <= new_col < 8 and 0 <= new_row < 8:
                target = board[new_row][new_col]
                if target != ' ' and ((piece.isupper() and target.islower()) or (piece.islower() and target.isupper())):
                    potential_moves.append((new_row, new_col))
        
        # En passant captures
        if en_passant_target != '-':
            ep_col = ord(en_passant_target[0]) - ord('a')
            ep_row = 8 - int(en_passant_target[1])
            if abs(col - ep_col) == 1 and row == (3 if piece == 'p' else 4):
                potential_moves.append((ep_row, ep_col))
    
    # Knight moves
    elif piece.lower() == 'n':
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]
        for drow, dcol in knight_moves:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target == ' ' or (piece.isupper() != target.isupper()):
                    potential_moves.append((new_row, new_col))
    
    # Bishop, Rook, and Queen moves
    directions = []
    if piece.lower() in 'bq':  # Bishop or Queen
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    if piece.lower() in 'rq':  # Rook or Queen
        directions += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for drow, dcol in directions:
        new_row, new_col = row + drow, col + dcol
        while 0 <= new_row < 8 and 0 <= new_col < 8:
            target = board[new_row][new_col]
            if target == ' ':
                potential_moves.append((new_row, new_col))
            elif (piece.isupper() != target.isupper()):
                potential_moves.append((new_row, new_col))
                break
            else:
                break
            new_row, new_col = new_row + drow, new_col + dcol
    
    # King moves (including castling)
    if piece.lower() == 'k':
        # Normal moves
        king_moves = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        for drow, dcol in king_moves:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = board[new_row][new_col]
                if target == ' ' or (piece.isupper() != target.isupper()):
                    potential_moves.append((new_row, new_col))
        
        # Castling
        if piece == 'K' and row == 7:
            # Kingside castling
            if 'K' in castling_rights and all(board[7][c] == ' ' for c in [5, 6]) and board[7][7] == 'R':
                if not any(is_square_attacked(board, 7, c, True) for c in [4, 5, 6]):
                    potential_moves.append((7, 6))
            # Queenside castling
            if 'Q' in castling_rights and all(board[7][c] == ' ' for c in [1, 2, 3]) and board[7][0] == 'R':
                if not any(is_square_attacked(board, 7, c, True) for c in [2, 3, 4]):
                    potential_moves.append((7, 2))
        elif piece == 'k' and row == 0:
            # Kingside castling
            if 'k' in castling_rights and all(board[0][c] == ' ' for c in [5, 6]) and board[0][7] == 'r':
                if not any(is_square_attacked(board, 0, c, False) for c in [4, 5, 6]):
                    potential_moves.append((0, 6))
            # Queenside castling
            if 'q' in castling_rights and all(board[0][c] == ' ' for c in [1, 2, 3]) and board[0][0] == 'r':
                if not any(is_square_attacked(board, 0, c, False) for c in [2, 3, 4]):
                    potential_moves.append((0, 2))
    
    # Filter out moves that would put or leave the king in check
    for move in potential_moves:
        if not would_move_cause_check(board, (row, col), move, turn, white_king_pos, black_king_pos):
            possible_moves.append(move)
    
    return possible_moves

def is_square_attacked(board, row, col, by_white):
    """Check if a square is attacked by any piece"""
    # Pawn attacks
    if by_white:
        if row > 0 and col > 0 and board[row-1][col-1] == 'P': return True
        if row > 0 and col < 7 and board[row-1][col+1] == 'P': return True
    else:
        if row < 7 and col > 0 and board[row+1][col-1] == 'p': return True
        if row < 7 and col < 7 and board[row+1][col+1] == 'p': return True
    
    # Knight attacks
    knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                   (1, -2), (1, 2), (2, -1), (2, 1)]
    knight = 'N' if by_white else 'n'
    for drow, dcol in knight_moves:
        new_row, new_col = row + drow, col + dcol
        if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == knight:
            return True
    
    # Bishop/Queen diagonal attacks
    bishop_queen = ['B', 'Q'] if by_white else ['b', 'q']
    for drow, dcol in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        new_row, new_col = row + drow, col + dcol
        while 0 <= new_row < 8 and 0 <= new_col < 8:
            piece = board[new_row][new_col]
            if piece != ' ':
                if piece in bishop_queen:
                    return True
                break
            new_row, new_col = new_row + drow, new_col + dcol
    
    # Rook/Queen straight attacks
    rook_queen = ['R', 'Q'] if by_white else ['r', 'q']
    for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + drow, col + dcol
        while 0 <= new_row < 8 and 0 <= new_col < 8:
            piece = board[new_row][new_col]
            if piece != ' ':
                if piece in rook_queen:
                    return True
                break
            new_row, new_col = new_row + drow, new_col + dcol
    
    # King attacks
    king = 'K' if by_white else 'k'
    for drow in [-1, 0, 1]:
        for dcol in [-1, 0, 1]:
            if drow == 0 and dcol == 0:
                continue
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == king:
                return True
    
    return False

def would_move_cause_check(board, start_pos, end_pos, turn, white_king_pos, black_king_pos):
    """Check if a move would leave the king in check"""
    # Make a copy of the board and king positions
    board_copy = [row[:] for row in board]
    white_king_pos_copy = white_king_pos
    black_king_pos_copy = black_king_pos
    
    # Perform the move on the copy
    piece = board_copy[start_pos[0]][start_pos[1]]
    board_copy[end_pos[0]][end_pos[1]] = piece
    board_copy[start_pos[0]][start_pos[1]] = " "
    
    # Update king position if king is moving
    if piece == 'K':
        white_king_pos_copy = end_pos
    elif piece == 'k':
        black_king_pos_copy = end_pos
    
    # Check if the move would leave the king in check
    king_pos = white_king_pos_copy if turn == "white" else black_king_pos_copy
    return is_square_attacked(board_copy, king_pos[0], king_pos[1], turn == "black")

def perform_move(board, start_pos, end_pos, turn, white_king_pos, black_king_pos, castling_rights, en_passant_target, halfmove_clock, fullmove_number, promotion_piece=None):
    start_row, start_col = start_pos
    end_row, end_col = end_pos
    piece = board[start_row][start_col]
    captured_piece = board[end_row][end_col]
    new_en_passant_target = "-"  # Reset en passant target by default
    is_capture = captured_piece != " "
    
    # Make a copy of the board to test for check
    board_copy = [row[:] for row in board]
    board_copy[end_row][end_col] = piece
    board_copy[start_row][start_col] = " "
    
    # Update king position for check testing
    test_white_king_pos = end_pos if piece == 'K' else white_king_pos
    test_black_king_pos = end_pos if piece == 'k' else black_king_pos
    
    # Test if opponent's king is in check after this move
    is_check = is_square_attacked(
        board_copy,
        test_black_king_pos[0] if turn == "white" else test_white_king_pos[0],
        test_black_king_pos[1] if turn == "white" else test_white_king_pos[1],
        turn == "white"
    )
    
    # Handle pawn moves and en passant first
    if piece.lower() == 'p':
        # Check for pawn double move
        if abs(start_row - end_row) == 2:
            passed_square_row = (start_row + end_row) // 2
            new_en_passant_target = chr(end_col + ord('a')) + str(8 - passed_square_row)
        
        # Handle en passant capture
        elif en_passant_target != '-':
            ep_col = ord(en_passant_target[0]) - ord('a')
            ep_row = 8 - int(en_passant_target[1])
            if end_col == ep_col and end_row == ep_row:
                captured_row = end_row + (1 if piece == 'P' else -1)
                board[captured_row][end_col] = " "
                is_capture = True
        
        # Handle promotion
        if end_row == 0 or end_row == 7:
            if not promotion_piece:
                promotion_piece = 'Q' if piece.isupper() else 'q'
    
    # Record move in algebraic notation
    is_capture = is_capture or (piece.lower() == 'p' and abs(end_col - start_col) == 1 and board[end_row][end_col] == " ")
    notation = get_algebraic_notation(
        start_pos, end_pos, piece,
        is_capture=is_capture,
        is_check=is_check,
        is_promotion=(piece.lower() == 'p' and (end_row == 0 or end_row == 7)),
        promotion_piece=promotion_piece
    )
    algebraic_moves.append(notation)
    
    # Make the actual move
    board[end_row][end_col] = promotion_piece if promotion_piece else piece
    board[start_row][start_col] = " "
    
    # Handle castling
    if piece.lower() == 'k' and abs(end_col - start_col) == 2:
        rook_row = start_row
        if end_col == 2:  # Queenside
            board[rook_row][0] = " "
            board[rook_row][3] = "R" if piece == "K" else "r"
        else:  # Kingside
            board[rook_row][7] = " "
            board[rook_row][5] = "R" if piece == "K" else "r"
    
    # Update king positions
    new_white_king_pos = end_pos if piece == 'K' else white_king_pos
    new_black_king_pos = end_pos if piece == 'k' else black_king_pos
    
    # Update castling rights
    new_castling_rights = castling_rights
    if piece == 'K':
        new_castling_rights = new_castling_rights.replace('K', '').replace('Q', '')
    elif piece == 'k':
        new_castling_rights = new_castling_rights.replace('k', '').replace('q', '')
    elif piece == 'R':
        if start_pos == (7, 0): new_castling_rights = new_castling_rights.replace('Q', '')
        elif start_pos == (7, 7): new_castling_rights = new_castling_rights.replace('K', '')
    elif piece == 'r':
        if start_pos == (0, 0): new_castling_rights = new_castling_rights.replace('q', '')
        elif start_pos == (0, 7): new_castling_rights = new_castling_rights.replace('k', '')
    
    if not new_castling_rights:
        new_castling_rights = '-'
    
    # Update halfmove clock
    if piece.lower() == 'p' or captured_piece != ' ':
        new_halfmove_clock = 0
    else:
        new_halfmove_clock = halfmove_clock + 1
    
    # Check for checkmate
    next_turn = "black" if turn == "white" else "white"
    if is_checkmate(board, next_turn, new_white_king_pos, new_black_king_pos, new_castling_rights, new_en_passant_target):
        global message_text, message_color, message_start_time
        message_text = f"Checkmate! {turn.capitalize()} wins!"
        message_color = MESSAGE_COLOR_SUCCESS
        message_start_time = pygame.time.get_ticks()
    elif is_check:
        message_text = "Check!"
        message_color = MESSAGE_COLOR_SUCCESS
        message_start_time = pygame.time.get_ticks()
    
    # Save game after move
    save_game()
    
    return (board, new_white_king_pos, new_black_king_pos, new_castling_rights, 
            new_en_passant_target, new_halfmove_clock, fullmove_number)

def handle_board_click(pos, width, height):
    """Handle clicks on the chess board"""
    global selected_piece, selected_piece_pos, possible_moves, turn, white_king_pos, black_king_pos
    global castling_rights, en_passant_target, halfmove_clock, fullmove_number, awaiting_promotion
    global promotion_start, promotion_end, board
    
    SQUARE_SIZE = min(width, height - UI_AREA_HEIGHT) // 8
    board_start_x = (width - (SQUARE_SIZE * 8)) // 2
    x, y = pos
    
    # Convert click coordinates to board position
    if y >= height - UI_AREA_HEIGHT:
        return
    
    col = (x - board_start_x) // SQUARE_SIZE
    row = y // SQUARE_SIZE
    
    if not (0 <= row < 8 and 0 <= col < 8):
        return
    
    # Handle promotion choice
    if awaiting_promotion and promotion_squares:
        for i, piece in enumerate(PROMOTION_PIECES):
            rect = pygame.Rect(
                board_start_x + promotion_end[1] * SQUARE_SIZE,
                i * PROMOTION_RECT_SIZE,
                PROMOTION_RECT_SIZE,
                PROMOTION_RECT_SIZE
            )
            if rect.collidepoint(x, y):
                result = perform_move(
                    board, promotion_start, promotion_end,
                    turn, white_king_pos, black_king_pos,
                    castling_rights, en_passant_target,
                    halfmove_clock, fullmove_number,
                    piece
                )
                if result:
                    board, white_king_pos, black_king_pos, castling_rights, en_passant_target, halfmove_clock, fullmove_number = result
                    turn = "black" if turn == "white" else "white"
                    if turn == "white":
                        fullmove_number += 1
                    save_game_state()  # Save state after successful move
                awaiting_promotion = False
                promotion_squares = None
                selected_piece = None
                selected_piece_pos = None
                possible_moves = []
                return
        return
    
    # Select piece or make move
    if selected_piece is None:
        piece = board[row][col]
        if piece != " " and ((turn == "white" and piece.isupper()) or (turn == "black" and piece.islower())):
            selected_piece = piece
            selected_piece_pos = (row, col)
            possible_moves = calculate_possible_moves(board, row, col, turn, white_king_pos, black_king_pos, castling_rights, en_passant_target)
            possible_moves = [(r, c) for r, c in possible_moves 
                            if not would_move_cause_check(board, (row, col), (r, c), turn, white_king_pos, black_king_pos)]
    else:
        if (row, col) in possible_moves:
            result = perform_move(
                board, selected_piece_pos, (row, col),
                turn, white_king_pos, black_king_pos,
                castling_rights, en_passant_target,
                halfmove_clock, fullmove_number
            )
            if result is None:  # Pawn promotion needed
                awaiting_promotion = True
                promotion_start = selected_piece_pos
                promotion_end = (row, col)
                promotion_squares = []
                for i, piece in enumerate(PROMOTION_PIECES):
                    rect = pygame.Rect(
                        board_start_x + col * SQUARE_SIZE,
                        i * PROMOTION_RECT_SIZE,
                        PROMOTION_RECT_SIZE,
                        PROMOTION_RECT_SIZE
                    )
                    promotion_squares.append((rect, piece))
            elif result:
                board, white_king_pos, black_king_pos, castling_rights, en_passant_target, halfmove_clock, fullmove_number = result
                turn = "black" if turn == "white" else "white"
                if turn == "white":
                    fullmove_number += 1
                save_game_state()  # Save state after successful move
        selected_piece = None
        selected_piece_pos = None
        possible_moves = []

def draw_board(screen, width, height):
    """Draw the chess board and pieces"""
    SQUARE_SIZE = min(width, height - UI_AREA_HEIGHT) // 8
    board_start_x = (width - (SQUARE_SIZE * 8)) // 2
    
    # Draw squares
    for row in range(8):
        for col in range(8):
            x = board_start_x + col * SQUARE_SIZE
            y = row * SQUARE_SIZE
            color = BOARD_COLOR_LIGHT if (row + col) % 2 == 0 else BOARD_COLOR_DARK
            pygame.draw.rect(screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
            
            # Highlight selected piece and possible moves
            if selected_piece_pos and selected_piece_pos == (row, col):
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                s.set_alpha(128)
                s.fill((255, 255, 0))
                screen.blit(s, (x, y))
            elif (row, col) in possible_moves:
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                s.set_alpha(128)
                s.fill((0, 255, 0))
                screen.blit(s, (x, y))
            
            # Draw piece
            piece = board[row][col]
            if piece != " ":
                draw_piece(screen, piece, x, y)
    
    # Draw promotion options if awaiting promotion
    if awaiting_promotion and promotion_squares:
        for rect, piece in promotion_squares:
            pygame.draw.rect(screen, (200, 200, 200), rect)
            draw_promotion_piece(screen, piece, rect)

def draw_piece(screen, piece, x, y):
    """Draw a piece on the board at the given position"""
    if piece == " ":
        return
        
    if piece in piece_images:
        # Draw the piece image
        screen.blit(piece_images[piece], (x, y))
    else:
        # Fall back to text representation
        text_surface = fallback_font.render(piece, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
        screen.blit(text_surface, text_rect)

def draw_promotion_piece(screen, piece, rect):
    """Draw a promotion piece option"""
    if piece in piece_images:
        image = piece_images[piece]
        image_rect = image.get_rect(center=rect.center)
        screen.blit(image, image_rect)
    else:
        text = fallback_font.render(piece.upper(), True, TEXT_COLOR)
        text_rect = text.get_rect(center=rect.center)
        screen.blit(text, text_rect)

def draw_ui(screen, width, height, current_time):
    """Draw the UI elements"""
    ui_area_y = height - UI_AREA_HEIGHT
    pygame.draw.rect(screen, (240, 240, 240), (0, ui_area_y, width, UI_AREA_HEIGHT))
    
    # Draw turn indicator
    turn_text = turn_font.render(f"{turn.capitalize()}'s turn", True, TEXT_COLOR)
    turn_rect = turn_text.get_rect(topleft=(10, ui_area_y + 10))
    screen.blit(turn_text, turn_rect)
    
    # Calculate button widths and spacing
    button_width = 70
    button_spacing = 10
    x_pos = turn_rect.right + 20

    # Draw moves button
    moves_button_rect = pygame.Rect(x_pos, ui_area_y + 10, button_width, BUTTON_HEIGHT)
    button_color = BUTTON_HOVER_COLOR if button_hovered and moves_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, moves_button_rect)
    moves_text = text_font.render("Moves", True, TEXT_COLOR)
    moves_text_rect = moves_text.get_rect(center=moves_button_rect.center)
    screen.blit(moves_text, moves_text_rect)
    x_pos += button_width + button_spacing

    # Draw undo button
    undo_button_rect = pygame.Rect(x_pos, ui_area_y + 10, button_width, BUTTON_HEIGHT)
    button_color = BUTTON_HOVER_COLOR if button_hovered and undo_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, undo_button_rect)
    undo_text = text_font.render("Undo", True, TEXT_COLOR)
    undo_text_rect = undo_text.get_rect(center=undo_button_rect.center)
    screen.blit(undo_text, undo_text_rect)
    x_pos += button_width + button_spacing

    # Draw redo button
    redo_button_rect = pygame.Rect(x_pos, ui_area_y + 10, button_width, BUTTON_HEIGHT)
    button_color = BUTTON_HOVER_COLOR if button_hovered and redo_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, redo_button_rect)
    redo_text = text_font.render("Redo", True, TEXT_COLOR)
    redo_text_rect = redo_text.get_rect(center=redo_button_rect.center)
    screen.blit(redo_text, redo_text_rect)
    x_pos += button_width + button_spacing

    # Draw copy FEN button
    copy_fen_button_rect = pygame.Rect(x_pos, ui_area_y + 10, button_width, BUTTON_HEIGHT)
    button_color = BUTTON_HOVER_COLOR if button_hovered and copy_fen_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, copy_fen_button_rect)
    copy_text = text_font.render("Copy", True, TEXT_COLOR)
    copy_text_rect = copy_text.get_rect(center=copy_fen_button_rect.center)
    screen.blit(copy_text, copy_text_rect)
    x_pos += button_width + button_spacing

    # Draw paste FEN button
    paste_fen_button_rect = pygame.Rect(x_pos, ui_area_y + 10, button_width, BUTTON_HEIGHT)
    button_color = BUTTON_HOVER_COLOR if button_hovered and paste_fen_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, paste_fen_button_rect)
    paste_text = text_font.render("Paste", True, TEXT_COLOR)
    paste_text_rect = paste_text.get_rect(center=paste_fen_button_rect.center)
    screen.blit(paste_text, paste_text_rect)
    x_pos += button_width + button_spacing

    # Draw FEN input box
    input_box_width = width - x_pos - 20
    input_box_rect = pygame.Rect(x_pos, ui_area_y + 10, input_box_width, BUTTON_HEIGHT)
    pygame.draw.rect(screen, (255, 255, 255), input_box_rect)
    pygame.draw.rect(screen, TEXT_COLOR if input_box_active else (200, 200, 200), input_box_rect, 2)

    # Draw current FEN or input text
    if input_box_active:
        text_to_display = fen_text
    else:
        text_to_display = board_to_full_fen(board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number)
    
    if text_to_display:
        fen_surface = text_font.render(text_to_display, True, TEXT_COLOR)
        fen_rect = fen_surface.get_rect(topleft=(input_box_rect.left + 5, input_box_rect.top + (BUTTON_HEIGHT - fen_surface.get_height()) // 2))
        
        # Create a clipping rect for the text
        clip_rect = pygame.Rect(input_box_rect.left + 5, input_box_rect.top,
                              input_box_rect.width - 10, input_box_rect.height)
        screen.set_clip(clip_rect)
        screen.blit(fen_surface, fen_rect)
        screen.set_clip(None)
    
    # Draw message if needed
    if message_text and current_time - message_start_time < MESSAGE_DURATION:
        message_surface = text_font.render(message_text, True, message_color)
        message_rect = message_surface.get_rect(center=(width // 2, ui_area_y + UI_AREA_HEIGHT - 20))
        screen.blit(message_surface, message_rect)
    
    # Draw moves popup if active
    popup_rects = None
    if show_moves_popup:
        popup_rects = draw_moves_popup(screen, width, height)
    
    return moves_button_rect, popup_rects, undo_button_rect, redo_button_rect, copy_fen_button_rect, paste_fen_button_rect, input_box_rect

def handle_ui_click(pos, current_time):
    """Handle clicks in the UI area"""
    global input_box_active, fen_text, message_text, message_color, message_start_time
    x, y = pos

    if input_box_rect.collidepoint(x, y):
        input_box_active = True
        if not fen_text:
            fen_text = board_to_full_fen(board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number)
    else:
        input_box_active = False

    if copy_fen_button_rect.collidepoint(x, y):
        try:
            current_fen = board_to_full_fen(board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number)
            pyperclip.copy(current_fen)
            message_text = "Position copied to clipboard"
            message_color = MESSAGE_COLOR_SUCCESS
            message_start_time = current_time
        except Exception as e:
            message_text = "Error copying to clipboard"
            message_color = MESSAGE_COLOR_ERROR
            message_start_time = current_time

    elif paste_fen_button_rect.collidepoint(x, y):
        try:
            clipboard_text = pyperclip.paste().strip()
            success, message = load_position_from_fen(clipboard_text)
            if success:
                fen_text = clipboard_text
                message_color = MESSAGE_COLOR_SUCCESS
                save_game()
            else:
                message_color = MESSAGE_COLOR_ERROR
            message_text = message
            message_start_time = current_time
        except Exception as e:
            message_text = "Error pasting from clipboard"
            message_color = MESSAGE_COLOR_ERROR
            message_start_time = current_time

def draw_moves_popup(screen, width, height):
    """Draw the moves history popup"""
    if not show_moves_popup:
        return None

    # Draw popup background
    popup_width = 300
    popup_height = 400
    popup_x = (width - popup_width) // 2
    popup_y = (height - popup_height) // 2
    pygame.draw.rect(screen, (255, 255, 255), (popup_x, popup_y, popup_width, popup_height))
    pygame.draw.rect(screen, (0, 0, 0), (popup_x, popup_y, popup_width, popup_height), 2)

    # Draw title
    title_text = title_font.render("Move History", True, TEXT_COLOR)
    title_rect = title_text.get_rect(centerx=popup_x + popup_width//2, top=popup_y + 10)
    screen.blit(title_text, title_rect)

    # Draw close button
    close_button_rect = pygame.Rect(popup_x + popup_width - 30, popup_y + 10, 20, 20)
    pygame.draw.rect(screen, (200, 200, 200), close_button_rect)
    close_text = text_font.render("×", True, TEXT_COLOR)
    close_text_rect = close_text.get_rect(center=close_button_rect.center)
    screen.blit(close_text, close_text_rect)

    # Draw moves list
    moves_surface = pygame.Surface((popup_width - 40, popup_height - 120))
    moves_surface.fill((255, 255, 255))
    y_offset = 0
    for i in range(0, len(algebraic_moves), 2):
        move_number = i // 2 + 1
        white_move = algebraic_moves[i]
        black_move = algebraic_moves[i + 1] if i + 1 < len(algebraic_moves) else ""
        move_text = f"{move_number}. {white_move} {black_move}"
        move_surface = text_font.render(move_text, True, TEXT_COLOR)
        moves_surface.blit(move_surface, (10, y_offset))
        y_offset += 25

    screen.blit(moves_surface, (popup_x + 20, popup_y + 40))

    # Draw buttons at the bottom
    button_y = popup_y + popup_height - 50
    button_width = 80
    button_height = 30
    button_spacing = 20

    # Copy button
    copy_button_rect = pygame.Rect(popup_x + (popup_width - 2*button_width - button_spacing) // 2,
                                 button_y, button_width, button_height)
    button_color = BUTTON_HOVER_COLOR if button_hovered and copy_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, copy_button_rect)
    copy_text = text_font.render("Copy", True, TEXT_COLOR)
    copy_text_rect = copy_text.get_rect(center=copy_button_rect.center)
    screen.blit(copy_text, copy_text_rect)

    # Reset button
    reset_button_rect = pygame.Rect(copy_button_rect.right + button_spacing,
                                  button_y, button_width, button_height)
    button_color = BUTTON_HOVER_COLOR if button_hovered and reset_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
    pygame.draw.rect(screen, button_color, reset_button_rect)
    reset_text = text_font.render("Reset", True, TEXT_COLOR)
    reset_text_rect = reset_text.get_rect(center=reset_button_rect.center)
    screen.blit(reset_text, reset_text_rect)

    return close_button_rect, copy_button_rect, reset_button_rect

def handle_moves_popup_click(pos, close_button_rect, copy_moves_rect):
    """Handle clicks on the moves popup"""
    global show_moves_popup, message_text, message_color, message_start_time
    
    if close_button_rect and close_button_rect.collidepoint(pos):
        show_moves_popup = False
    elif copy_moves_rect and copy_moves_rect.collidepoint(pos):
        # Create PGN-style move list
        moves_text = ""
        for i in range(0, len(algebraic_moves), 2):
            move_num = f"{i//2 + 1}. "
            white_move = algebraic_moves[i] if i < len(algebraic_moves) else ""
            black_move = f" {algebraic_moves[i+1]}" if i+1 < len(algebraic_moves) else ""
            moves_text += f"{move_num}{white_move}{black_move} "
        
        try:
            pyperclip.copy(moves_text.strip())
            message_text = "Moves copied to clipboard"
            message_color = MESSAGE_COLOR_SUCCESS
            message_start_time = pygame.time.get_ticks()
        except Exception as e:
            message_text = "Error copying moves"
            message_color = MESSAGE_COLOR_ERROR
            message_start_time = pygame.time.get_ticks()

def handle_input_events(event, mouse_pos):
    """Handle input-related events"""
    global input_box_active, fen_text, message_text, message_color, message_start_time, button_hovered
    global board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number
    
    if event.type == pygame.MOUSEBUTTONDOWN:
        # Handle input box click
        if input_box_rect.collidepoint(mouse_pos):
            input_box_active = True
            # Initialize fen_text with current position if empty
            if not fen_text:
                fen_text = board_to_full_fen(board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number)
        else:
            input_box_active = False
            
        # Handle paste button click
        if paste_button_rect and paste_button_rect.collidepoint(mouse_pos):
            try:
                # Get text from clipboard
                clipboard_text = pyperclip.paste().strip()
                print(f"Pasting FEN: {clipboard_text}")  # Debug message
                
                # Try to load the position
                success, message = load_position_from_fen(clipboard_text)
                if success:
                    fen_text = clipboard_text  # Update text field with valid FEN
                    message_color = MESSAGE_COLOR_SUCCESS
                else:
                    message_color = MESSAGE_COLOR_ERROR
                message_text = message
                message_start_time = pygame.time.get_ticks()
            except Exception as e:
                print(f"Paste error: {str(e)}")  # Debug message
                message_text = "Error pasting from clipboard"
                message_color = MESSAGE_COLOR_ERROR
                message_start_time = pygame.time.get_ticks()
                
        # Handle copy button click
        elif copy_button_rect and copy_button_rect.collidepoint(mouse_pos):
            try:
                # Generate current FEN and copy to clipboard
                current_fen = board_to_full_fen(board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number)
                print(f"Copying FEN: {current_fen}")  # Debug message
                pyperclip.copy(current_fen)
                message_text = "Position copied to clipboard"
                message_color = MESSAGE_COLOR_SUCCESS
                message_start_time = pygame.time.get_ticks()
            except Exception as e:
                print(f"Copy error: {str(e)}")  # Debug message
                message_text = "Error copying to clipboard"
                message_color = MESSAGE_COLOR_ERROR
                message_start_time = pygame.time.get_ticks()
    
    elif event.type == pygame.KEYDOWN and input_box_active:
        handle_keyboard_input(event)
    
    # Update button hover state
    button_hovered = (paste_button_rect and paste_button_rect.collidepoint(mouse_pos)) or \
                    (copy_button_rect and copy_button_rect.collidepoint(mouse_pos)) or \
                    (undo_button_rect and undo_button_rect.collidepoint(mouse_pos)) or \
                    (redo_button_rect and redo_button_rect.collidepoint(mouse_pos))

def board_to_full_fen(board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number):
    """Convert the current board state to FEN notation"""
    # Generate board part
    fen = ""
    for row in board:
        empty_count = 0
        for square in row:
            if square == " ":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += square
        if empty_count > 0:
            fen += str(empty_count)
        fen += "/"
    fen = fen[:-1]  # Remove the trailing /

    # Add turn
    turn_char = 'w' if turn == "white" else 'b'
    
    # Add castling rights
    if not castling_rights or castling_rights == '-':
        castling_rights = '-'
    
    # Ensure en passant target is valid
    if not en_passant_target or en_passant_target == '-':
        en_passant_target = '-'
    
    # Ensure move counters are valid integers
    halfmove_clock = max(0, int(halfmove_clock))
    fullmove_number = max(1, int(fullmove_number))
    
    # Combine all parts
    fen += f" {turn_char} {castling_rights} {en_passant_target} {halfmove_clock} {fullmove_number}"
    return fen

def load_position_from_fen(fen):
    """Load a position from FEN notation"""
    global board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number, white_king_pos, black_king_pos
    
    try:
        # Split FEN into its components
        parts = fen.strip().split(' ')
        if len(parts) != 6:
            raise ValueError("Invalid FEN: must have 6 components")
        
        position, turn_char, castle_rights, en_passant, half_moves, full_moves = parts
        
        # Parse board position
        new_board = []
        rows = position.split('/')
        if len(rows) != 8:
            raise ValueError("Invalid FEN: board must have 8 ranks")
        
        for row in rows:
            board_row = []
            col = 0
            for char in row:
                if char.isdigit():
                    empty_count = int(char)
                    board_row.extend([" "] * empty_count)
                    col += empty_count
                else:
                    if char not in piece_placeholders:
                        raise ValueError(f"Invalid piece character: {char}")
                    board_row.append(char)
                    # Track king positions
                    if char == 'K':
                        white_king_pos = (len(new_board), col)
                    elif char == 'k':
                        black_king_pos = (len(new_board), col)
                    col += 1
            if col != 8:
                raise ValueError(f"Invalid FEN: rank {len(new_board) + 1} has wrong number of squares")
            new_board.append(board_row)
        
        # Validate turn
        if turn_char not in ['w', 'b']:
            raise ValueError("Invalid turn: must be 'w' or 'b'")
        
        # Validate castling rights
        if castle_rights != '-' and not all(c in 'KQkq' for c in castle_rights):
            raise ValueError("Invalid castling rights")
        
        # Validate en passant target
        if en_passant != '-':
            if len(en_passant) != 2 or not ('a' <= en_passant[0] <= 'h' and '1' <= en_passant[1] <= '8'):
                raise ValueError("Invalid en passant target")
        
        # Validate move counters
        try:
            half_moves = int(half_moves)
            full_moves = int(full_moves)
            if half_moves < 0 or full_moves < 1:
                raise ValueError()
        except ValueError:
            raise ValueError("Invalid move counters")
        
        # Update game state
        board = new_board
        turn = "white" if turn_char == 'w' else "black"
        castling_rights = castle_rights
        en_passant_target = en_passant
        halfmove_clock = half_moves
        fullmove_number = full_moves
        
        return True, "Position loaded successfully"
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error loading position: {str(e)}"

def get_algebraic_notation(start_pos, end_pos, piece, is_capture, is_check=False, is_promotion=False, promotion_piece=None):
    """Convert a move to algebraic notation"""
    start_row, start_col = start_pos
    end_row, end_col = end_pos
    
    # Get the piece letter (empty for pawns)
    piece_letter = piece.upper() if piece.upper() not in ['P', 'K'] else ''
    if piece.upper() == 'K' and abs(end_col - start_col) == 2:
        # Castling
        return "O-O" if end_col > start_col else "O-O-O"
    
    # Get the destination square
    dest = chr(end_col + ord('a')) + str(8 - end_row)
    
    # Add capture symbol
    capture_symbol = 'x' if is_capture else ''
    
    # Add source square for pawns making captures
    if piece.upper() == 'P' and is_capture:
        source = chr(start_col + ord('a'))
        notation = f"{source}{capture_symbol}{dest}"
    else:
        notation = f"{piece_letter}{capture_symbol}{dest}"
    
    # Add promotion piece
    if is_promotion and promotion_piece:
        notation += f"={promotion_piece.upper()}"
    
    # Add check symbol
    if is_check:
        notation += "+"
    
    return notation

def is_checkmate(board, turn, white_king_pos, black_king_pos, castling_rights, en_passant_target):
    """Check if the current position is checkmate"""
    # First check if king is in check
    king_pos = white_king_pos if turn == "white" else black_king_pos
    if not is_square_attacked(board, king_pos[0], king_pos[1], turn == "black"):
        return False
    
    # Try all possible moves for all pieces
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            # Check only pieces of the current player
            if (turn == "white" and piece.isupper()) or (turn == "black" and piece.islower()):
                moves = calculate_possible_moves(board, row, col, turn, white_king_pos, black_king_pos, castling_rights, en_passant_target)
                # Filter moves that would still leave king in check
                valid_moves = [(r, c) for r, c in moves 
                             if not would_move_cause_check(board, (row, col), (r, c), turn, white_king_pos, black_king_pos)]
                if valid_moves:
                    return False
    return True

def save_game():
    """Save current game state to file"""
    state = {
        'board': board,
        'turn': turn,
        'castling_rights': castling_rights,
        'en_passant_target': en_passant_target,
        'halfmove_clock': halfmove_clock,
        'fullmove_number': fullmove_number,
        'white_king_pos': white_king_pos,
        'black_king_pos': black_king_pos,
        'move_history': move_history,
        'algebraic_moves': algebraic_moves,
        'current_move_index': current_move_index
    }
    try:
        with open(SAVE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Error saving game: {e}")

def load_game():
    """Load game state from file"""
    global board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number
    global white_king_pos, black_king_pos, move_history, algebraic_moves, current_move_index
    
    try:
        with open(SAVE_FILE, 'r') as f:
            state = json.load(f)
            board = state['board']
            turn = state['turn']
            castling_rights = state['castling_rights']
            en_passant_target = state['en_passant_target']
            halfmove_clock = state['halfmove_clock']
            fullmove_number = state['fullmove_number']
            white_king_pos = tuple(state['white_king_pos'])
            black_king_pos = tuple(state['black_king_pos'])
            move_history = state['move_history']
            algebraic_moves = state['algebraic_moves']
            current_move_index = state['current_move_index']
            return True
    except Exception as e:
        print(f"Error loading game: {e}")
        return False

def reset_game():
    """Reset the game to initial state"""
    global board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number
    global white_king_pos, black_king_pos, move_history, algebraic_moves, current_move_index
    global message_text, message_color, message_start_time
    
    # Reset board to initial position
    board = [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        [" ", " ", " ", " ", " ", " ", " ", " "],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]
    
    # Reset game state
    turn = "white"
    castling_rights = "KQkq"
    en_passant_target = "-"
    halfmove_clock = 0
    fullmove_number = 1
    white_king_pos = (7, 4)
    black_king_pos = (0, 4)
    
    # Reset move history
    move_history = []
    algebraic_moves = []
    current_move_index = -1
    
    # Save initial state
    save_game()
    
    # Show message
    message_text = "Game reset"
    message_color = MESSAGE_COLOR_SUCCESS
    message_start_time = pygame.time.get_ticks()

def handle_keyboard_input(event):
    """Handle keyboard input events"""
    global input_box_active, fen_text, message_text, message_color, message_start_time

    if event.key == pygame.K_RETURN:
        success, message = load_position_from_fen(fen_text)
        message_color = MESSAGE_COLOR_SUCCESS if success else MESSAGE_COLOR_ERROR
        message_text = message
        message_start_time = pygame.time.get_ticks()
        if success:
            input_box_active = False
            save_game()
    elif event.key == pygame.K_BACKSPACE:
        fen_text = fen_text[:-1]
    elif event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL:
        try:
            clipboard_text = pyperclip.paste().strip()
            success, message = load_position_from_fen(clipboard_text)
            if success:
                fen_text = clipboard_text
                message_color = MESSAGE_COLOR_SUCCESS
                save_game()
            else:
                message_color = MESSAGE_COLOR_ERROR
            message_text = message
            message_start_time = pygame.time.get_ticks()
        except Exception as e:
            message_text = "Error pasting from clipboard"
            message_color = MESSAGE_COLOR_ERROR
            message_start_time = pygame.time.get_ticks()
    else:
        fen_text += event.unicode

def main():
    """Main game loop"""
    global width, height, screen, input_box_active, fen_text, message_text, message_color, message_start_time
    global board, turn, castling_rights, en_passant_target, halfmove_clock, fullmove_number
    global show_moves_popup, undo_button_rect, redo_button_rect, input_box_rect, copy_fen_button_rect, paste_fen_button_rect
    
    # Initialize pygame and fonts
    init_pygame()
    
    # Try to load saved game
    if os.path.exists(SAVE_FILE):
        if load_game():
            message_text = "Game loaded"
            message_color = MESSAGE_COLOR_SUCCESS
            message_start_time = pygame.time.get_ticks()
    else:
        # Save initial state
        save_game()
    
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        mouse_pos = pygame.mouse.get_pos()
        
        # Clear screen
        screen.fill((255, 255, 255))
        
        # Draw board and UI
        draw_board(screen, width, height)
        moves_button_rect, popup_rects, undo_button_rect, redo_button_rect, copy_fen_button_rect, paste_fen_button_rect, input_box_rect = draw_ui(screen, width, height, current_time)
        
        # Update display
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                
                if show_moves_popup and popup_rects:
                    close_button_rect, copy_button_rect, reset_button_rect = popup_rects
                    if close_button_rect.collidepoint(x, y):
                        show_moves_popup = False
                    elif copy_button_rect.collidepoint(x, y):
                        copy_moves_to_clipboard()
                    elif reset_button_rect.collidepoint(x, y):
                        reset_game()
                        show_moves_popup = False
                elif moves_button_rect.collidepoint(x, y):
                    show_moves_popup = True
                elif undo_button_rect.collidepoint(x, y):
                    undo_move()
                elif redo_button_rect.collidepoint(x, y):
                    redo_move()
                elif y < height - UI_AREA_HEIGHT:
                    handle_board_click(event.pos, width, height)
                else:
                    handle_ui_click(event.pos, pygame.time.get_ticks())
            elif event.type == pygame.KEYDOWN and input_box_active:
                handle_keyboard_input(event)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()