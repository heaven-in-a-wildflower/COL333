import time
import math
import random
import numpy as np
from helper import *
from copy import deepcopy

def update_board(state, move):
    board = state[0]
    player_num = state[1]
    row, col = move
    if board[row, col] == 0:
        board[row, col] = player_num
        return (board, (player_num%2)+1)
    else:
        err = 'Invalid move by player {}. Column {}'.format(player_num, move)
        raise Exception(err)

def is_game_over(state, move, actions=None):
    opponent = state[1]%2 + 1
    if (move):
        if check_win(state[0], move, opponent)[0]:
            return opponent
        if actions==None:
            actions = get_valid_actions(state[0])
        if actions:
            return 0
        else:
            return -1
    return 0

class MCTSNode4:
    ph_value = None
    ph_n = None
    start = None
    player_number = None

    @classmethod
    def pv(cls, move):
        row, col = move
        return cls.ph_value[row, col]

    @classmethod
    def pn(cls, move):
        row, col = move
        return cls.ph_n[row, col]

    def __init__(self, state, player_number=None, parent=None, parent_action=None):
        self.state = state
        self.turn = state[1]
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._value = 0
        self._untried_actions = get_valid_actions(self.state[0])

        if (parent == None):
            MCTSNode4.ph_value = np.zeros([state[0].shape[0], state[0].shape[0]]).astype(np.int32)
            MCTSNode4.ph_n = np.zeros([state[0].shape[0], state[0].shape[0]]).astype(np.int32)
            MCTSNode4.start = state
            MCTSNode4.player_number = player_number

        return

    def n(self):
        return self._number_of_visits

    def v(self):
        return self._value

    def expand(self):
        action = self._untried_actions.pop()
        next_state = update_board(self.state, action)
        child_node = MCTSNode4(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return (is_game_over(self.state, self.parent_action, self._untried_actions) != 0)

    def rollout(self):
        current_rollout_state = self.state
        old_action = None
        action = self.parent_action
        result = is_game_over(current_rollout_state, action)
        if result != 0:
            return (1 if result == MCTSNode4.player_number else -1) 
        
        while True:
            possible_moves = get_valid_actions(current_rollout_state[0])
            if len(possible_moves) == 0:
                print(current_rollout_state[0])
                raise Exception("No possible moves available")
            new_action = self.rollout_policy(possible_moves, action, old_action)
            old_action = action
            action = new_action
            current_rollout_state = update_board(current_rollout_state, action)
            result = is_game_over(current_rollout_state, action)
            if result != 0:
                break
        return (1 if result == MCTSNode4.player_number else -1)

    def rollout_policy(self, possible_moves, action, old_action):
        # return possible_moves[np.random.randint(len(possible_moves))]

        padosis = get_neighbours(self.state[0].shape[0], action)
        if (old_action):
            padosis.extend(get_neighbours(self.state[0].shape[0], old_action))
        dic = {}
        padosi_ke_padosi = []
        for padosi in padosis:
            x = get_neighbours(self.state[0].shape[0], padosi)
            for i in x:
                if i in dic:
                    dic[i] += 1
                else:
                    dic[i] = 1
        weights = [1 if move not in dic else dic[move] for move in possible_moves]
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        return random.choices(possible_moves, weights=normalized_weights, k=1)[0]

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._value += (result if (self.turn != MCTSNode4.player_number) else result*(-1))
        if self.parent:
            row, col = self.parent_action
            MCTSNode4.ph_n[row, col] += 1
            MCTSNode4.ph_value[row, col] += (result if (self.turn != MCTSNode4.player_number) else result*(-1))
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return (len(self._untried_actions) == 0)

    def best_child(self, c_param=0.9, w_param=10):
        k = np.log(self.n())
        choices_weights = ([(c.v() / c.n()) + c_param * np.sqrt((k / c.n()))
                            +(MCTSNode4.pv(c.parent_action)/MCTSNode4.pn(c.parent_action))*(w_param/(c.n()-c.v()+1))
                            for c in self.children])
        return self.children[np.argmax(choices_weights)]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            elif self.parent_action == None and current_node.n() < 1000:
                old_state = current_node.state
                current_node = current_node.children[np.random.randint(len(current_node.children))]
                current_node.state = update_board(old_state, current_node.parent_action)
            elif current_node.n() < 100:
                old_state = current_node.state
                current_node = current_node.children[np.random.randint(len(current_node.children))]
                current_node.state = update_board(old_state, current_node.parent_action)
            else:
                old_state = current_node.state
                current_node = current_node.best_child()
                current_node.state = update_board(old_state, current_node.parent_action)
        return current_node

    def best_score_child(self):
        choices_weights = ([(c.v() / c.n()) for c in self.children])
        return self.children[np.argmax(choices_weights)]

    def best_action(self):
        simulation_no = 10000
        for i in range(simulation_no):
            self.state = (MCTSNode4.start[0].copy(), MCTSNode4.start[1])
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        # for c in self.children:
        #     print(c.parent_action,c.v(),c.n())

        mera_beta = self.best_score_child()
        mera_beta_score = mera_beta.v()/mera_beta.n()
        # print(mera_beta.parent_action, mera_beta_score)
        return mera_beta.parent_action

class UnionFind:
    def __init__(self, dim):
        self.parent = {(i, j):(i, j) for i in range(dim) for j in range(dim)}
        self.rank = {(i, j):0 for i in range(dim) for j in range(dim)}
        self.reach_corners = {(i, j):0 for i in range(dim) for j in range(dim)}
        self.reach_edges = {(i, j):0 for i in range(dim) for j in range(dim)}
        self.player = {(i, j):0 for i in range(dim) for j in range(dim)}
        self.dim = dim
        self.corners = get_all_corners(dim)
        for pos in range(len(self.corners)):
            self.reach_corners[self.corners[pos]] = (1 << pos)
        self.edges = [set(edge) for edge in get_all_edges(dim)]
        for pos in range(len(self.edges)):
            for x in self.edges[pos]:
                self.reach_edges[x] = (1 << pos)

    def find(self, x):  
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
                self.reach_corners[root_y] |= self.reach_corners[root_x]
                self.reach_edges[root_y] |= self.reach_edges[root_x]
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
                self.reach_corners[root_x] |= self.reach_corners[root_y]
                self.reach_edges[root_x] |= self.reach_edges[root_y]
            else:
                self.parent[root_y] = root_x
                self.reach_corners[root_x] |= self.reach_corners[root_y]
                self.reach_edges[root_x] |= self.reach_edges[root_y]
                self.rank[root_x] += 1

def update_dsu(dsu, move, player_num):
    dsu.player[move] = player_num

    neighbours = get_neighbours(dsu.dim, move)
    for padosi in neighbours:
        if dsu.player[padosi] == player_num:
            dsu.union(move, padosi)


def update_board6(state, move, dsu):
    board = state[0]
    player_num = state[1]
    row, col = move
    if board[row, col] == 0:
        board[row, col] = player_num
        update_dsu(dsu, move, player_num)
        return (board, (player_num%2)+1)
    else:
        err = 'Invalid move by player {}. Column {}'.format(player_num, move)
        raise Exception(err)

def get_correct_neighbours(dim: int, vertex: Tuple[int, int]) -> List[Tuple[int, int]]:
    i, j = vertex
    siz = dim//2
    neighbours = []
    if i > 0 and j <= siz and j > 0:
        neighbours.append((i - 1, j - 1))
    if j > 0:
        neighbours.append((i, j - 1))
    if j > siz and i < dim - 1:
        neighbours.append((i + 1, j - 1))
    if i < dim - 1:
        neighbours.append((i + 1, j))
    if i > 0 and j >= siz and j < dim - 1:
        neighbours.append((i - 1, j + 1))
    if j < siz and i < dim - 1:
        neighbours.append((i + 1, j + 1))
    if j < dim - 1:
        neighbours.append((i, j + 1))
    if i > 0:
        neighbours.append((i - 1, j))
    return neighbours

def check_winner(board: np.array, move: Tuple[int, int], player_num: int, dsu: UnionFind) -> bool:
    root = dsu.find(move)
    if dsu.reach_corners[root].bit_count() >= 2:
        # print("BRIDGE")
        return True
    if dsu.reach_edges[root].bit_count() >= 3:
        # print("FORK")
        return True
    neighbours = get_correct_neighbours(dsu.dim, move)
    s = ""
    for padosi in neighbours:
        if dsu.player[padosi] == dsu.player[move]:
            s+='1'
        else:
            s+='0'
    connected = s.count('1')
    s+=s
    if connected<2:
        return False
    # print("CHECK RING")
    player_board = (board == player_num)
    if check_ring(player_board, move):
        return True
    # print("----------")
    return False

def is_game_over6(state, move, dsu, actions=None):
    opponent = state[1]%2 + 1
    if (move):
        if check_winner(state[0], move, opponent, dsu):
            return opponent
        if actions==None:
            actions = get_valid_actions(state[0])
        if actions:
            return 0
        else:
            return -1
    return 0

def get_valid_actions2(board: np.array, prev_untried_moves: set, move: Tuple[int, int]) -> set:
    dim = board.shape[0]

    def is_within_bounds(i, j):
        return 0 <= i < dim and 0 <= j < dim

    prev_untried_moves.discard(move)
    row, col = move
    for i in range(row - 2, row + 3):
        for j in range(col - 2, col + 3):
            if is_within_bounds(i, j) and board[i, j] == 0:
                prev_untried_moves.add((i, j))
    return prev_untried_moves

def root_actions(board: np.array) -> set[Tuple[int, int]]:
    dim = board.shape[0]

    def is_within_bounds(i, j):
        return 0 <= i < dim and 0 <= j < dim

    valid_moves = set()
    occupied_positions = np.argwhere((board == 1) | (board == 2))
    for pos in occupied_positions:
        row, col = pos
        for i in range(row - 2, row + 3):
            for j in range(col - 2, col + 3):
                if is_within_bounds(i, j) and board[i, j] == 0:
                    valid_moves.add((i, j))

    return valid_moves


class MCTSNode:
    ph_value = None
    ph_n = None
    start = None
    player_number = None
    dsu = None
    dsu_start = None
    init_valid_actions = None
    num_iter = 5000

    @classmethod
    def pv(cls, move):
        row, col = move
        return cls.ph_value[row, col]

    @classmethod
    def pn(cls, move):
        row, col = move
        return cls.ph_n[row, col]

    def __init__(self, state, player_number=None, parent=None, parent_action=None, valid_moves=None, num_iter=None):
        self.state = state
        self.turn = state[1]
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._value = 0
        self.valid_moves = valid_moves
        self._untried_actions = None

        if (parent == None):
            MCTSNode.ph_value = np.zeros([state[0].shape[0], state[0].shape[0]]).astype(np.int32)
            MCTSNode.ph_n = np.zeros([state[0].shape[0], state[0].shape[0]]).astype(np.int32)
            MCTSNode.start = state
            MCTSNode.player_number = player_number
            MCTSNode.dsu = UnionFind(state[0].shape[0])
            for i in range(state[0].shape[0]):
                for j in range(state[0].shape[0]):
                    x = state[0][i,j]
                    if (x==1 or x==2):
                        update_dsu(MCTSNode.dsu, (i,j), x)
            MCTSNode.dsu_start = deepcopy(MCTSNode.dsu)

            self.valid_moves = root_actions(state[0])
            MCTSNode.init_valid_actions = deepcopy(self.valid_moves)
        
        self._untried_actions = list(self.valid_moves)

        if (num_iter):
            MCTSNode.num_iter = num_iter
            print("Iter changed to ", num_iter)
        return

    def n(self):
        return self._number_of_visits

    def v(self):
        return self._value

    def expand(self):
        action = self._untried_actions.pop()
        next_state = update_board6(self.state, action, MCTSNode.dsu)
        valid_moves = get_valid_actions2(next_state[0], self.valid_moves, action)
        child_node = MCTSNode(next_state, parent=self, parent_action=action, valid_moves=valid_moves)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return (is_game_over6(self.state, self.parent_action, MCTSNode.dsu, self._untried_actions) != 0)

    def rollout(self):
        current_rollout_state = self.state
        old_action = None
        action = self.parent_action
        possible_moves = self.valid_moves
        result = is_game_over6(current_rollout_state, action, MCTSNode.dsu)
        if result != 0:
            return (1 if result == MCTSNode.player_number else -1) 
        
        while True:
            if len(possible_moves) == 0:
                print(current_rollout_state[0])
                raise Exception("No possible moves available")
            new_action = self.rollout_policy(list(possible_moves), action, old_action)
            old_action = action
            action = new_action
            current_rollout_state = update_board6(current_rollout_state, action, MCTSNode.dsu)
            possible_moves = get_valid_actions2(current_rollout_state[0], possible_moves, action)
            result = is_game_over6(current_rollout_state, action, MCTSNode.dsu)
            if result != 0:
                break

        return (1 if result == MCTSNode.player_number else -1)

    def rollout_policy(self, possible_moves, action, old_action):
        # return possible_moves[np.random.randint(len(possible_moves))]

        padosis = get_neighbours(self.state[0].shape[0], action)
        if (old_action):
            padosis.extend(get_neighbours(self.state[0].shape[0], old_action))
        dic = {}
        padosi_ke_padosi = []
        for padosi in padosis:
            x = get_neighbours(self.state[0].shape[0], padosi)
            for i in x:
                if i in dic:
                    dic[i] += 1
                else:
                    dic[i] = 1
        weights = [1 if move not in dic else dic[move] for move in possible_moves]
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        return random.choices(possible_moves, weights=normalized_weights, k=1)[0]

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._value += (result if (self.turn != MCTSNode.player_number) else result*(-1))
        if self.parent:
            row, col = self.parent_action
            MCTSNode.ph_n[row, col] += 1
            MCTSNode.ph_value[row, col] += (result if (self.turn != MCTSNode.player_number) else result*(-1))
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return (len(self._untried_actions) == 0)

    def best_child(self, c_param=0.9, w_param=10):
        k = np.log(self.n())
        choices_weights = ([(c.v() / c.n()) + c_param * np.sqrt((k / c.n()))
                            +(MCTSNode.pv(c.parent_action)/MCTSNode.pn(c.parent_action))*(w_param/(c.n()-c.v()+1))
                            for c in self.children])
        return self.children[np.argmax(choices_weights)]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            elif self.parent_action == None and current_node.n() < 500:
                old_node = current_node
                current_node = current_node.children[np.random.randint(len(current_node.children))]
                current_node.state = update_board6(old_node.state, current_node.parent_action, MCTSNode.dsu)
                current_node.valid_moves = get_valid_actions2(current_node.state[0], old_node.valid_moves, current_node.parent_action)
            elif current_node.n() < 100:
                old_node = current_node
                current_node = current_node.children[np.random.randint(len(current_node.children))]
                current_node.state = update_board6(old_node.state, current_node.parent_action, MCTSNode.dsu)
                current_node.valid_moves = get_valid_actions2(current_node.state[0], old_node.valid_moves, current_node.parent_action)
            else:
                old_node = current_node
                current_node = current_node.best_child()
                current_node.state = update_board6(old_node.state, current_node.parent_action, MCTSNode.dsu)
                current_node.valid_moves = get_valid_actions2(current_node.state[0], old_node.valid_moves, current_node.parent_action)
        return current_node

    def best_score_child(self):
        choices_weights = ([(c.v() / c.n()) for c in self.children])
        return self.children[np.argmax(choices_weights)]

        # children_sorted = sorted(self.children, key=lambda g: (g.v()/g.n(), g.n()), reverse=True)
        # return children_sorted[0]


    def best_action(self):
        simulation_no = MCTSNode.num_iter
        for i in range(simulation_no):
            self.state = (MCTSNode.start[0].copy(), MCTSNode.start[1])
            self.valid_moves = deepcopy(MCTSNode.init_valid_actions)
            # print(self.valid_moves)
            MCTSNode.dsu = deepcopy(MCTSNode.dsu_start)
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        # for c in self.children:
        #     print(c.parent_action,c.v(),c.n())

        mera_beta = self.best_score_child()
        mera_beta_score = mera_beta.v()/mera_beta.n()
        # print(mera_beta.parent_action, mera_beta_score)
        return mera_beta.parent_action

def differ_by_one(arr1, arr2):
    if arr1.shape != arr2.shape:
        return False

    differences = np.sum(arr1 != arr2)

    return differences <= 1

class AIPlayer:

    def __init__(self, player_number: int, timer):
        
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.prev_time = 10000
        self.time_ = 10000
        self.num_iter = 5000

    def check_forced_move(self, board):
        dim = board.shape[0]
        siz = dim // 2
        moves = set()
        for i in range(dim):
            for j in range(dim):
                if board[i][j] == 1 or board[i][j] == 2:
                    if i > 0 and board[i - 1][j] == 0:
                        moves.add((i - 1, j))
                    if i < dim - 1 and board[i + 1][j] == 0:
                        moves.add((i + 1, j))
                    if j > 0 and board[i][j - 1] == 0:
                        moves.add((i, j - 1))
                    if j < dim - 1 and board[i][j + 1] == 0:
                        moves.add((i, j + 1))
                    if i > 0 and j <= siz and j > 0 and board[i - 1][j - 1] == 0:
                        moves.add((i - 1, j - 1))
                    if i > 0 and j >= siz and j < dim - 1 and board[i - 1][j + 1] == 0:
                        moves.add((i - 1, j + 1))
                    if j < siz and i < dim - 1 and board[i + 1][j + 1] == 0:
                        moves.add((i + 1, j + 1))
                    if j > siz and i < dim - 1 and board[i + 1][j - 1] == 0:
                        moves.add((i + 1, j - 1))

        player_num = self.player_number
        for move in moves:
            board[move[0]][move[1]] = player_num
            win, _ = check_win(board, move, player_num)
            board[move[0]][move[1]] = 3 - player_num
            loss, _ = check_win(board, move, 3 - player_num)
            board[move[0]][move[1]] = 0
            if win:
                return move
            elif loss:
                return move
        
        return None


    def get_move(self, state: np.array) -> Tuple[int, int]:

        if state.shape[0] == 7:
            s0 = np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 3],
                [3, 3, 0, 0, 0, 3, 3],
                [3, 3, 3, 0, 3, 3, 3]
            ])

            if np.array_equal(state, s0):
                return (0,0)

            for i in get_all_corners(7):
                s1 = deepcopy(s0)
                s1[0,0] = 1
                if np.array_equal(state, s1):
                    return (1,2)
                elif differ_by_one(state, s1):
                    if (state[1,2] == 0):
                        return (1,2)
                    elif (state[2,1] == 0):
                        return (2,1)

                s2 = deepcopy(s0)
                s2[0,3] = 1
                if np.array_equal(state, s2):
                    return (1,2)
                elif differ_by_one(state, s2):
                    if (state[1,2] == 0):
                        return (1,2)
                    elif (state[1,4] == 0):
                        return (1,4)

                s3 = deepcopy(s0)
                s3[0,6] = 1
                if np.array_equal(state, s3):
                    return (1,4)
                elif differ_by_one(state, s2):
                    if (state[2,5] == 0):
                        return (2,5)
                    elif (state[1,4] == 0):
                        return (1,4)

                s4 = deepcopy(s0)
                s4[3,6] = 1
                if np.array_equal(state, s4):
                    return (2,5)
                elif differ_by_one(state, s2):
                    if (state[2,5] == 0):
                        return (2,5)
                    elif (state[4,4] == 0):
                        return (4,4)

                s5 = deepcopy(s0)
                s5[6,3] = 1
                if np.array_equal(state, s5):
                    return (4,4)
                elif differ_by_one(state, s2):
                    if (state[4,2] == 0):
                        return (4,2)
                    elif (state[4,4] == 0):
                        return (4,4)

                s6 = deepcopy(s0)
                s6[3,0] = 1
                if np.array_equal(state, s6):
                    return (4,2)
                elif differ_by_one(state, s2):
                    if (state[4,2] == 0):
                        return (4,2)
                    elif (state[2,1] == 0):
                        return (2,1)


        if state.shape[0] == 11:
            s0 = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                [3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3],
                [3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3],
                [3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 3],
                [3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3]
            ])

            if np.array_equal(state, s0):
                return (0,0)

            for i in get_all_corners(7):
                s1 = deepcopy(s0)
                s1[0,0] = 1
                if np.array_equal(state, s1):
                    return (0,5)
                s2 = deepcopy(s0)
                s2[0,5] = 1
                if np.array_equal(state, s2):
                    return (0,10)
                s3 = deepcopy(s0)
                s3[0,10] = 1
                if np.array_equal(state, s3):
                    return (5,10)
                s4 = deepcopy(s0)
                s4[5,10] = 1
                if np.array_equal(state, s4):
                    return (10,5)
                s5 = deepcopy(s0)
                s5[10,5] = 1
                if np.array_equal(state, s5):
                    return (5,0)
                s6 = deepcopy(s0)
                s6[5,0] = 1
                if np.array_equal(state, s6):
                    return (0,0)

        forced_move = self.check_forced_move(state)
        if forced_move:
            return forced_move

        alert = False

        if state.shape[0] == 11:
            self.prev_time = self.time_
            self.time_ = fetch_remaining_time(self.timer, self.player_number)
            for k in range(1,5):
                if self.prev_time > 100*k and self.time_ < 100*k:
                    self.num_iter = 1000*(k+1)

            root = MCTSNode(state = (state,self.player_number), player_number=self.player_number, num_iter=self.num_iter)
        else:
            root = MCTSNode4(state = (state,self.player_number), player_number=self.player_number)
        return root.best_action()

        raise NotImplementedError('Whoops I don\'t know what to do')

