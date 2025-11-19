def alphabeta(board, depth, alpha, beta, maximizing_player):
    if game_over(board):
        return evaluate(board)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in get_available_moves(board):
            result = alphabeta(make_move(board, move, 'X'), depth + 1, alpha, beta, False)
            max_eval = max(max_eval, result)
            alpha = max(alpha, result)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_available_moves(board):
            result = alphabeta(make_move(board, move, 'O'), depth + 1, alpha, beta, True)
            min_eval = min(min_eval, result)
            beta = min(beta, result)
            if beta <= alpha:
                break
        return min_eval
