# app.py - 完整后端程序
from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room, leave_room, emit
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# 存储房间信息
rooms = {}

def get_initial_board():
    """标准跳棋初始棋盘（8x8）"""
    board = [[0] * 8 for _ in range(8)]
    # 玩家1（红方）初始位置
    for row in range(3):
        start = 0 if row % 2 == 0 else 1
        for col in range(start, 8, 2):
            board[row][col] = 1
    # 玩家2（蓝方）初始位置
    for row in range(5, 8):
        start = 1 if row % 2 == 0 else 0
        for col in range(start, 8, 2):
            board[row][col] = 2
    return board

def is_valid_position(x, y):
    return 0 <= x < 8 and 0 <= y < 8

def get_all_valid_moves(board, player, from_pos=None):
    """
    获取所有合法移动（包含连续跳跃）
    :param from_pos: 用于处理连续跳跃时的起始位置
    """
    x, y = from_pos if from_pos else (None, None)
    moves = []

    if from_pos is None:
        # 查找所有己方棋子
        for i in range(8):
            for j in range(8):
                if board[i][j] == player:
                    moves += get_jump_moves(board, player, (i, j), True)
        return moves

    # 处理连续跳跃
    return get_jump_moves(board, player, (x, y), False)

def get_jump_moves(board, player, from_pos, include_simple):
    x, y = from_pos
    directions = [
        (-1, -1), (-1, 1),  # 红方移动方向（向上）
        (1, -1), (1, 1)      # 蓝方移动方向（向下）
    ] if player == 1 else [
        (1, -1), (1, 1),     # 蓝方移动方向（向下）
        (-1, -1), (-1, 1)    # 红方移动方向（向上）
    ]

    moves = []
    # 简单移动（相邻位置）
    if include_simple:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid_position(nx, ny) and board[nx][ny] == 0:
                moves.append((nx, ny))

    # 跳跃移动（递归查找连续跳跃）
    jump_paths = deque()
    jump_paths.append(([from_pos], []))  # (路径, 被跳过的棋子)

    while jump_paths:
        path, jumped = jump_paths.popleft()
        last_x, last_y = path[-1]

        for dx, dy in directions:
            # 检查中间棋子
            mid_x = last_x + dx
            mid_y = last_y + dy
            # 检查目标位置
            target_x = last_x + 2 * dx
            target_y = last_y + 2 * dy

            if not is_valid_position(target_x, target_y):
                continue

            if (mid_x, mid_y) in jumped:
                continue  # 不能重复跳过同一个棋子

            if board[mid_x][mid_y] not in [0, player] and board[target_x][target_y] == 0:
                new_path = path + [(target_x, target_y)]
                new_jumped = jumped + [(mid_x, mid_y)]
                jump_paths.append((new_path, new_jumped))
                moves.append((target_x, target_y))

    return moves

def validate_move(room, from_pos, to_pos, player):
    board = rooms[room]['board']
    # 检查是否移动己方棋子
    if board[from_pos[0]][from_pos[1]] != player:
        return False

    # 获取所有合法移动
    all_moves = get_all_valid_moves(board, player, from_pos)
    return (to_pos[0], to_pos[1]) in all_moves

def check_winner(room):
    board = rooms[room]['board']
    # 检查玩家1是否全部到达对方区域
    player1_win = all(cell != 1 for row in board[:5] for cell in row)
    # 检查玩家2是否全部到达对方区域
    player2_win = all(cell != 2 for row in board[3:] for cell in row)

    # 检查是否有合法移动
    player1_moves = len(get_all_valid_moves(board, 1)) > 0
    player2_moves = len(get_all_valid_moves(board, 2)) > 0

    if player1_win or not player2_moves:
        return 0  # 玩家1胜
    elif player2_win or not player1_moves:
        return 1  # 玩家2胜
    return -1

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('join')
def handle_join(data):
    room = data['room']
    if room not in rooms:
        rooms[room] = {'players': [], 'board': get_initial_board(), 'current_player': 1}

    if len(rooms[room]['players']) >= 2:
        emit('error', {'message': '房间已满'})
        return

    join_room(room)
    rooms[room]['players'].append(request.sid)

    # 分配玩家颜色
    player = len(rooms[room]['players']) - 1
    emit('assign_color', {'color': player}, room=request.sid)

    if len(rooms[room]['players']) == 2:
        # 通知双方游戏开始
        socketio.emit('game_start', {
            'board': rooms[room]['board'],
            'current_player': rooms[room]['current_player']
        }, room=room)

@socketio.on('move')
def handle_move(data):
    room = data['room']
    from_pos = tuple(data['from'])
    to_pos = tuple(data['to'])
    player = data['player']

    if not validate_move(room, from_pos, to_pos, player):
        emit('invalid_move', {'message': '非法移动'}, room=request.sid)
        return

    # 执行移动
    board = rooms[room]['board']
    board[to_pos[0]][to_pos[1]] = player
    board[from_pos[0]][from_pos[1]] = 0

    # 处理被跳过的棋子移除
    dx = (to_pos[0] - from_pos[0]) // 2
    dy = (to_pos[1] - from_pos[1]) // 2
    if abs(dx) == 1 and abs(dy) == 1:  # 简单移动
        pass
    else:  # 跳跃移动
        mid_x = from_pos[0] + dx
        mid_y = from_pos[1] + dy
        board[mid_x][mid_y] = 0

    # 检查是否还有连续跳跃
    has_more_jumps = len(get_jump_moves(board, player, to_pos, False)) > 0

    # 更新游戏状态
    if not has_more_jumps:
        rooms[room]['current_player'] = 1 if player == 2 else 2

    # 广播更新
    socketio.emit('update_board', {
        'board': board,
        'current_player': rooms[room]['current_player']
    }, room=room)

    # 检查胜负
    winner = check_winner(room)
    if winner != -1:
        socketio.emit('game_over', {'winner': winner}, room=room)

@socketio.on('get_moves')
def handle_get_moves(data):
    room = data['room']
    from_pos = tuple(data['from'])
    player = data['player']
    board = rooms[room]['board']
    moves = get_all_valid_moves(board, player, from_pos)
    emit('highlight_moves', moves, room=request.sid)

if __name__ == '__main__':
    socketio.run(app, debug=True)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)