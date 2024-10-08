import time
import math
import copy
import random
import numpy as np
from helper import *
from collections import deque
from typing import List, Tuple, Dict, Union


initial_player = 0 
opponent_move=(-1,-1)
moves_done=0
exp_time=0
# Create a 2D array of empty dictionaries
neighbor_grid = np.empty((11, 11), dtype=object)
for i in range(11):
    for j in range(11):
        neighbor_grid[i, j] = {}

# Create a 2D array of zeros
bonus_grid = np.zeros((11, 11), dtype=float)
opp_bonus_grid = np.zeros((11, 11), dtype=float)
global_state = np.zeros((11, 11), dtype=int)

# Precompute all edge and corner distances
all_edge_distances = np.zeros((11, 11, 6), dtype=int)
all_corner_distances = np.zeros((11, 11, 6), dtype=int)

occupied_corners = set()
occupied_edges = set()


def bfs_distance(start, targets):
    queue = deque([(start, 0)])
    visited = np.zeros((11, 11), dtype=bool)
    visited[start] = True
    
    while queue:
        (x, y), dist = queue.popleft()
        
        if (x, y) in targets:
            return dist
        
        for nx, ny in get_neighbours(11, (x, y)):
            if is_valid_position((nx, ny)) and not visited[nx, ny]:
                visited[nx, ny] = True
                queue.append(((nx, ny), dist + 1))
    
    return float('inf')  # If no path is found

def precompute_all_distances():
    global all_edge_distances, all_corner_distances
    corners = [(0, 0), (0, 10), (5, 10), (10, 5), (0, 5), (5, 0)]
    edges = [
        [(0, i) for i in range(1,5)],  # Top edge
        [(0, i) for i in range(5,10)],  # Top-right edge
        [(i, 10) for i in range(1,5)],  # Right edge
        [(5+i, 10-i) for i in range(1,5)],  # Bottom-right edge
        [(5+i, i) for i in range(1,5)],  # Bottom-left edge
        [(i, 0) for i in range(1,5)]  # Left edge
    ]
    
    for i in range(11):
        for j in range(11):
            if is_valid_position((i, j)):
                # Edge distances
                for edge_idx, edge in enumerate(edges):
                    all_edge_distances[i, j, edge_idx] = bfs_distance((i, j), edge)
                
                # Corner distances
                for corner_idx, corner in enumerate(corners):
                    all_corner_distances[i, j, corner_idx] = bfs_distance((i, j), [corner])

def update_occupied_corners_and_edges(action, player):
        global occupied_corners,occupied_edges
        i, j = action
        corners = [(0, 0), (0, 10), (5, 10), (10, 5), (10, 0), (5, 0)]
        if (i, j) in corners:
            occupied_corners.add(corners.index((i, j)))
        
        edges = [
            [(0, i) for i in range(1,5)],  # Top edge
            [(0, i) for i in range(5,10)],  # Top-right edge
            [(i, 10) for i in range(1,5)],  # Right edge
            [(5+i, 10-i) for i in range(1,5)],  # Bottom-right edge
            [(5+i, i) for i in range(1,5)],  # Bottom-left edge
            [(i, 0) for i in range(1,5)]  # Left edge
        ]
        
        for edge_idx, edge in enumerate(edges):
            if (i, j) in edge and player == initial_player:
                occupied_edges.add(edge_idx)

def get_min_distances(action):
        unoccupied_edges = set(range(6)) - occupied_edges
        unoccupied_corners = set(range(6)) - occupied_corners
        
        if unoccupied_edges:
            min_edge_dist = min(all_edge_distances[action[0], action[1], idx] for idx in unoccupied_edges)
        else:
            min_edge_dist = 0  # All edges are occupied
        
        if unoccupied_corners:
            min_corner_dist = min(all_corner_distances[action[0], action[1], idx] for idx in unoccupied_corners)
        else:
            min_corner_dist = 0  # All corners are occupied
        
        return min_edge_dist, min_corner_dist

def precompute_neighbors():
    global neighbor_grid
    global bonus_grid
    global opp_bonus_grid
    for i in range(11):
        for j in range(11):
            # print(i,j)
            neighbor_grid[i, j] = get_all_neighbors(i, j)
            # print(neighbor_grid[i,j])
    bonus_grid = np.zeros((11, 11), dtype=float)
    opp_bonus_grid = np.zeros((11, 11), dtype=float)
    
    for i in range(11):
        for j in range(11):
            action_=(i,j)
            if (global_state[i,j]==initial_player):
                update_bonus_grid(action_,initial_player)
            elif (global_state[i,j]==3-initial_player):
                update_bonus_grid(action_,3-initial_player)

def get_all_neighbors(i, j):
    first_neighbors = []
    second_vc_neighbors = []
    second_non_vc_neighbors = []
    
    temp=get_neighbours(11,(i,j))
    for ni, nj in temp:
        if is_valid_position((ni, nj)):
            first_neighbors.append((ni, nj))
            
            for nni, nnj in get_neighbours(11,(ni,nj)):
                if is_valid_position((nni, nnj)) and (nni, nnj) != (i, j) and (nni,nnj) not in temp:
                    check1,check2=is_virtual_connection((i, j), (nni, nnj))
                    # print(i,j,nni,nnj)
                    # print(check1,check2)
                    if check1:
                        second_vc_neighbors.append((nni, nnj))
                    if check2:
                        second_non_vc_neighbors.append((nni, nnj))
    return {
        'first': first_neighbors,
        'second_vc': list(set(second_vc_neighbors)),
        'second_non_vc': list(set(second_non_vc_neighbors))
    }

def update_bonus_grid(action,player):
        global bonus_grid
        global opp_bonus_grid
        i, j = action
        bonus_grid[i, j]=0
        opp_bonus_grid[i, j]=0
        if player==initial_player:
            for ni, nj in neighbor_grid[(i, j)]['first']:
                bonus_grid[ni, nj] = max(bonus_grid[ni, nj], 0.9)#0.9
            for ni, nj in neighbor_grid[(i, j)]['second_vc']:
                bonus_grid[ni, nj] = max(bonus_grid[ni, nj], 1.5)
            for ni, nj in neighbor_grid[(i, j)]['second_non_vc']:
                bonus_grid[ni, nj] = max(bonus_grid[ni, nj], 0.4)

        elif player==3-initial_player:
            for ni, nj in neighbor_grid[(i, j)]['first']:
                opp_bonus_grid[ni, nj] = max(opp_bonus_grid[ni, nj], 1)
            for ni, nj in neighbor_grid[(i, j)]['second_vc']:
                opp_bonus_grid[ni, nj] = max(opp_bonus_grid[ni, nj], 0.2)
            for ni, nj in neighbor_grid[(i, j)]['second_non_vc']:
                opp_bonus_grid[ni, nj] = max(opp_bonus_grid[ni, nj], 0.2)


def is_virtual_connection(move1,move2):
    neighbours1=get_neighbours(11,move1)
    neighbours1=set(neighbours1)

    # if move2 in virtual_positions:
    neighbours2=get_neighbours(11,move2)
    neighbours2=set(neighbours2)
    s1=neighbours1.intersection(neighbours2)
    our_cnt=0
    their_cnt=0
    for i in s1:
        if global_state[i]==3-initial_player:
            their_cnt+=1
        if global_state[i]==initial_player:
            our_cnt+=1

    if len(s1)==1:
        if their_cnt+our_cnt==0:
            return (False,True)
    if len(s1)==2:
        if their_cnt+our_cnt==0:
            return (True,False)
    return (False,False)

def is_valid_position(pos):
    i, j = pos
    return 0 <= i < 11 and 0 <= j < 11

class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size
        self.edge_corner_bits = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
            self.edge_corner_bits[x] = self.edge_corner_bits[self.parent[x]]
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.edge_corner_bits[root_y] |= self.edge_corner_bits[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.edge_corner_bits[root_x] |= self.edge_corner_bits[root_y]
        else:
            self.parent[root_y] = root_x
            self.edge_corner_bits[root_x] |= self.edge_corner_bits[root_y]
            self.rank[root_x] += 1

uf = UnionFind(11 * 11)

def debug_():
    for i in range(11):
        for j in range(11):
            print((uf.edge_corner_bits)[uf.find(i*11+j)],end=' ')
        print()
    print("parents: ")
    for i in range(11):
        for j in range(11):
            print((uf.parent)[i*11+j],end=' ')
        print()


def get_index(x: int, y: int, dim: int) -> int:
    return x * dim + y

def get_edge_corner_bit(x: int, y: int, dim: int) -> int:
    if x == 0 and y == 0:
        return 1 << 0  # Top-left corner
    elif x == 0 and y == dim - 1:
        return 1 << 1  # Top-right corner
    elif x == 0 and y == dim//2:
        return 1 << 2  # Top corner
    elif x == dim//2 and y == 0:
        return 1 << 3  # Bottom-left corner
    elif x == dim//2 and y == dim - 1:
        return 1 << 4  # Bottom-right corner
    elif x == dim-1 and y == dim//2:
        return 1 << 5  # Bottom corner


    elif x == 0 and y>0 and y<=dim//2-1:
        return 1 << 6  # Top left edge
    elif x == 0 and y<=dim-2 and y>=dim//2+1:
        return 1 << 7  # Top left edge
    elif y == 0 and x>0 and x<=dim//2-1:
        return 1 << 8  # Left edge
    elif x>0 and x<=dim//2-1 and y == dim - 1 :
        return 1 << 9  # Right edge
    elif x - y == dim//2 and x>dim//2 and x<dim-1:
        return 1 << 10 # Bottom left edge
    elif x + y==(dim-1)+(dim//2) and x>dim//2 and x<dim-1:
        return 1 << 11  #  Bottom right edge
    else:
        return 0
    
def perform_union(board, move):
    # Perform unions for the current move
    global uf
    x, y = move
    dim = 11
    move_idx = get_index(x, y, dim)
    
    new_bits = get_edge_corner_bit(x, y, 11)
    # print("IN perform union")
    # print(move,new_bits)
    # print(board)
    # print()
    for nx, ny in get_neighbours(dim, move):
        if is_valid(nx, ny, dim) and board[nx, ny] == board[x,y]:
            neighbor_idx = get_index(nx, ny, dim)
            new_bits |= uf.edge_corner_bits[uf.find(neighbor_idx)]
    # print("new_bits",new_bits)
    uf.edge_corner_bits[move_idx] = new_bits


    
    for nx, ny in get_neighbours(dim, move):
        if is_valid(nx, ny, dim) and board[nx, ny] == board[x,y]:
            neighbor_idx = get_index(nx, ny, dim)
            uf.union(move_idx, neighbor_idx)

def check_fork_and_bridge_(board: np.array, move: Tuple[int, int],player) -> Tuple[bool, str]:
    x, y = move
    dim = 11
    move_idx = get_index(x, y, dim)
    perform_union(board, move)
    
    # Check for fork or bridge
    root = uf.find(move_idx)
    bits = uf.edge_corner_bits[root]
    # print(board)
    # print(move)
    # print(bits)

    # Check for fork (at least 3 edges)
    edge_count = bin(bits & 0b111111000000).count('1')
    #print("edge_cnt: ",edge_count)
    
    if edge_count >= 3:
        return True, "fork"

    # Check for bridge (at least 2 corners)
    corner_count = bin(bits & 0b000000111111).count('1')
    #print("corner_count: ",corner_count)
    if corner_count >= 2:
        return True, "bridge"
    return False, None

def is_valid(x: int, y: int, dim: int) -> bool:
    return 0 <= x < dim and 0 <= y < dim

def check_win_(board: np.array, move: Tuple[int, int], player_num: int, path:List[Tuple[int, int]]=None) -> Tuple[bool, Union[str, None]]:
    win,way = check_fork_and_bridge_(board, move,player_num)
    if win:
        # print("winning.....",way)
        # print(board)
        # print(move)
        # debug_()
        return True
    my_board=(board==player_num)
    if check_ring(my_board, move):
        # print("Ring Win")
        # print(board)
        # print(my_board)
        # print(move)
        return True
    return False

class Node:
    def __init__(self, state, player, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 1
        self.value = 1
        self.player = player
        self.rave_value = 0
        self.rave_visits = 0
        self.score = 0.0
        # self.loc_bonus=0.0
        # self.dist_bonus=0.0
        # self.uct_bonus=0.0


    def is_fully_expanded(self):
        return len(self.children) == self.state.get_num_actions()

    def best_child(self, player, exploration_weight=1.2):
        choices_weights = [
            (child.value / child.visits) + exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        best_child = self.children[choices_weights.index(max(choices_weights))]
        return best_child

    def expand(self, sim):
        global uf,exp_time
        exp_time=0
        t1=time.time()
        # print("inside expand")
        # debug_()
        action = (0,0)
        # uf_copy0 = copy.deepcopy(uf)
        legal_actions = self.state.get_legal_actions()
        if sim<0:
            action = self.state.choose_action_with_bonus()
        else:
            action = random.choice(legal_actions)
        #debug_()
        #print(action)
        #self.children.append(child_node)
        # uf_copy = copy.deepcopy(uf)
        for action1 in legal_actions:
            # uf = copy.deepcopy(uf_copy0)
            next_state = self.state.take_action_without_uf(action1, self.player)
            child_node = Node(next_state, 3 - self.player, self)
            self.children.append(child_node)
        
        next_state = self.state.take_action(action, self.player)
        child_node = Node(next_state, 3 - self.player, self)

        # uf = copy.deepcopy(uf_copy)
        # print(next_state.grid)
        # debug_()
        # print("leaving expand")
        t2=time.time()
        exp_time+=t2-t1
        return child_node

    def update(self, winner):
        self.visits += 1
        if winner != 0:
            if winner == 3 - self.player:
                self.value += 1
            else:
                self.value -= 1
    
    def update_rave(self, action, result):
        self.rave_visits += 1
        self.rave_value += result

class MCTS:
    def __init__(self, state, exploration_weight=0.1, rave_constant=700, ppr_threshold=0.25, ppr_probability=0.95, min_rave_visits=15):
        self.exploration_weight = exploration_weight
        self.virtual_connection_cache = {}
        self.rave_constant = rave_constant
        self.ppr_threshold = ppr_threshold
        self.ppr_probability = ppr_probability
        self.min_rave_visits = min_rave_visits
        self.root = None
        self.ppr_cache = {}  # New cache for PPR actions
        self.root = Node(state,initial_player)

    def search(self, current_state, player, num_simulations=3000):
        # moves=2*moves_done+initial_player
        # if moves<6:
        #     num_simulations=20
        # elif moves<16:
        #     num_simulations=3500
        # else:
        #     num_simulations=3000
        global bonus_grid
        global opp_bonus_grid
        global occupied_corners, occupied_edges
        global neighbor_grid
        global uf
        for i in range(11):
            for j in range(11):
                neighbor_grid[i, j] = {}

        precompute_neighbors()
        # print(neighbor_grid)
        # print("---------------------------")
        init_bonus_grid=bonus_grid.copy()
        init_opp_bonus_grid=opp_bonus_grid.copy()
        init_occupied_corners = occupied_corners.copy()
        init_occupied_edges = occupied_edges.copy()
        uf_copy = copy.deepcopy(uf)

        
        # #print("init grid\n",init_bonus_grid)
        root = Node(current_state, player)
        init_root = Node(current_state, player)
        self.ppr_cache = {}
        # #print("entering search")
        #debug_()
        t_1,t_2,t_3=0,0,0
        for sim in range(num_simulations):
            bonus_grid=init_bonus_grid.copy()
            opp_bonus_grid=init_opp_bonus_grid.copy()
            uf = copy.deepcopy(uf_copy)
            occupied_edges=init_occupied_edges.copy()
            occupied_corners=init_occupied_corners.copy()
            player = initial_player
            root = init_root
            # #print("Entering selection")
            # print(current_state.grid)
            #debug_()
            t1=time.time()
            node = self.selection(root,player,sim)
            t2=time.time()
            t_1+=t2-t1
            # print(node.state.grid)
            #debug_()
            # print("Finished selection")
            winner, moves_played = self.simulation(node.state, node.player)
            t3=time.time()
            t_2+=t3-t2
            self.backpropagation(node, winner, moves_played)
            t4=time.time()
            t_3+=t4-t3
    
        print("total,sel,sim,bp: ",t_1+t_2+t_3,t_1,t_2,t_3)
        print("exp_time",exp_time)
        occupied_corners=init_occupied_corners.copy()
        occupied_edges=init_occupied_edges.copy()

        bonus_grid=init_bonus_grid.copy()
        uf = copy.deepcopy(uf_copy)
        opp_bonus_grid=init_opp_bonus_grid.copy()
        
        best_state=self.best_action(init_root)
        best_move = self.get_action_difference(init_root.state,best_state)
        update_occupied_corners_and_edges(best_move,initial_player)
        update_bonus_grid(best_move,initial_player)
        # print(best_state.grid)
        # print("IN SEARCH")
        # debug_()
        perform_union(best_state.grid,best_move)
        # print(bonus_grid)
        # debug_()
        # print("SEARCH over")
        return best_state
    
    def selection(self, node, player, sim):
        while True:
            if node.is_fully_expanded():
                best_child = self.best_uct_child(node)
                last_action = best_child.state.last_action
                if best_child.state.is_terminal(last_action, node.player):
                    return best_child
                node = best_child
            else:
                return node.expand(sim)
            
    
    def best_uct_child(self, node):
        best_score = float('-inf')
        best_child = None
        final_uct=0
        final_db=0
        final_lb=0
        best_child_bonus = 0
        moves=2*moves_done+initial_player
        for child in node.children:
            uct_score = self.uct_score(node, child)
            last_action = child.state.last_action
            my_local_bonus = child.state.get_bonus(last_action,initial_player)
            opp_local_bonus=opp_bonus_grid[last_action]
            # Get dynamic minimum distances
            edge_factor = 2000
            corner_factor = 1000

            if moves<16:
                # edge_factor = 350#400 done,500
                # corner_factor = 150#250,150 done
                
                edge_factor = 700
                corner_factor = 200
            else:
                edge_factor = 800
                corner_factor = 300

            edge_dist, corner_dist = get_min_distances(last_action)
            # print("Here: ",last_action,edge_dist, corner_dist)
            rem_edge_dist=(6-edge_dist)**2
            rem_corner_dist=(6-corner_dist)**2
            
            my_locality_factor = 20.0
            opp_loc_factor=20
            
            if moves<16:
                my_locality_factor=12 #11.5
                opp_loc_factor=8 #8
            else:
                my_locality_factor= 10#8,8,10
                opp_loc_factor= 12 #15,12,10
            # my_locality_factor=0.1
            local_bonus_score=my_local_bonus/my_locality_factor+opp_local_bonus/opp_loc_factor
            # local_bonus_score=my_local_bonus/my_locality_factor+opp_local_bonus/opp_loc_factor
            
            distance_bonus = rem_edge_dist/edge_factor + rem_corner_dist/corner_factor
            
            uct_factor=2.0
            if moves<10:
                uct_factor=1
            elif moves<16:
                uct_factor=1
            else:
                uct_factor=1
            uct_factor=1
            combined_score = uct_score/uct_factor + local_bonus_score + distance_bonus
            # print("last_action: ",last_action)
            # #print("scores: ",uct_score/uct_factor,local_bonus_score,distance_bonus,combined_score)
            
            #combined_score = uct_score

            if combined_score > best_score:
                # best_child_bonus = local_bonus_score
                best_score = combined_score
                best_child = child
                # final_uct=uct_score
                # final_db=distance_bonus
                # final_lb=local_bonus_score
        best_child.score = best_score
        # best_child.uct_bonus=final_uct
        # best_child.dist_bonus=final_db
        # best_child.loc_bonus=final_lb
        # #print("best_score: ",best_score,final_uct,final_lb,final_db,self.get_action_difference(node.state,best_child.state))
        return best_child
    
    def uct_score(self, parent, child):
        if child.visits == 0:
            return float('inf')
        
        exploitation = child.value / child.visits
        exploration = math.sqrt(2 * math.log(parent.visits) / child.visits)
        
        beta = child.rave_visits / (child.visits + child.rave_visits + 4 * self.rave_constant * child.visits * child.rave_visits)
        rave = beta * (child.rave_value / child.rave_visits if child.rave_visits > 0 else 0)
        
        return (1 - beta) * exploitation + beta * rave + self.exploration_weight * exploration

    
    def simulation(self, state, player):
        current_state = state
        current_player = player
        moves_played = []
        last_move = None
        move_count = 0
        #player_moves = {1: [], 2: []}  # Track moves for each player
        # print("inside simulation")
        # debug_()
        # print("Entering simulation")
        legal_actions = state.get_legal_actions()
        while len(legal_actions)>0:
            # PPR implementation
            idx = random.randint(0,len(legal_actions)-1)
            # Use cached PPR actions if available, otherwise compute and cache them
            state_hash = hash(state)
            if state_hash not in self.ppr_cache:
                self.ppr_cache[state_hash] = self.compute_ppr_actions(state, current_player)
            
            ppr_actions = self.ppr_cache[state_hash]
            
            if ppr_actions and random.random() < self.ppr_probability:
                action = random.choice(ppr_actions)
            else:
                action = legal_actions[idx]
                del legal_actions[idx]
            # print("Before")
            # print(current_state.grid)
            # debug_()
            new_state = current_state.take_action(action, current_player)
            moves_played.append((action, current_player))
            #player_moves[initial_player].append(action)
            move_count += 1

            # print("action: ",action)
            # print("player: ",current_player)
            # print(new_state.grid)
            # debug_()
            # print("After")
            if new_state.is_terminal(action, current_player):
                #print("finish simulation")
                return new_state.get_winner(action, current_player), moves_played

            current_state = new_state
            current_player = 3 - current_player

    def compute_ppr_actions(self, state, player):
        ppr_actions = []
        rave_stats = self.get_rave_stats(state, player)
        for action, (rave_value, rave_visits) in rave_stats.items():
            if rave_visits > 0 and rave_value / rave_visits > self.ppr_threshold:
                ppr_actions.append(action)
        return ppr_actions
    
    def get_rave_stats(self, state, player):
        node = self.find_node(state)
        rave_stats = {}
        while node is not None:
            if node.rave_visits >= self.min_rave_visits:
                for child in node.children:
                    action = child.state.last_action
                    #action=self.get_action_difference(node.state,child.state)
                    if action not in rave_stats:
                        rave_stats[action] = (child.rave_value, child.rave_visits)
            node = node.parent
        return rave_stats

    def find_node(self, state):
        node = self.root
        while node.children:
            found_child = False
            for child in node.children:
                if child.state == state:
                    node = child
                    found_child = True
                    break
            if not found_child:
                break
        return node if node.state == state else None
    
    def backpropagation(self, node, winner, moves_played):
        while node is not None:
            node.update(winner)
            for action, player in moves_played:
                if player == node.player:
                    result = 1 if winner == player else 0
                    node.update_rave(action, result)
            node = node.parent

    def best_action(self, node):
        factor = 0.5
        max_score = -float('inf')
        max_state = None
        max_visits = 0
        max_value = 0
        for child in node.children:
            # current_score =  factor * np.log(child.visits) + child.score
            current_score =  child.visits
            # print("curr_params: ",child.state.last_action,current_score, np.log(child.visits))
            # print("score: ",child.score,",uct: ",child.uct_bonus,",loc ",child.loc_bonus,",dist: ",child.dist_bonus)
            if current_score > max_score:
                max_score = current_score
                max_state = child.state
                max_visits = child.visits
                max_value = child.value
        # print("max_params: ",max_score, np.log(max_visits))
        return max_state
        #return max(node.children, key=lambda child: child.value/max(1,child.visits) + factor * np.log(child.visits)).state
    
    def get_action_difference(self, state1, state2):
        diff = np.argwhere(state1.grid != state2.grid)
        if len(diff) == 1:
            return tuple(diff[0])
        else:
            raise ValueError("Could not determine the action that led to the new state")

class State:
    def __init__(self, grid=None, action = None):
        if grid is None:
            self.grid = np.zeros((11, 11), dtype=int)  # Changed to 11x11 for Havannah
        else:
            self.grid = grid

        self.last_action = action
        # self.occupied_corners = set()
        # self.occupied_edges = set()
    
    def get_bonus(self, action,player):
        if action==(-1,-1):
            return 0
        if player==initial_player:
            ans=bonus_grid[action]
            return ans   
        if player==3-initial_player:
            return opp_bonus_grid[action]   

    def choose_action_with_bonus(self):
        legal_actions = self.get_legal_actions()
        bonus_scores = [bonus_grid[action] for action in legal_actions]
        total_bonus = sum(bonus_scores)
        if total_bonus == 0:
            return random.choice(legal_actions)
        else:
            probabilities = [score / total_bonus for score in bonus_scores]
            return random.choices(legal_actions, weights=probabilities, k=1)[0]

    def is_terminal(self, action, player):
        return self.check_winner(action, player) or self.is_full()

    def is_terminal_without_uf(self, action, player):
        return self.check_winner_without_uf(action, player) or self.is_full()

    def get_winner(self, action, player):
        if self.check_winner(action, player):
            return player
        return 0 

    def get_legal_actions(self):
        return [tuple(action) for action in np.argwhere(self.grid == 0)]
    
    def take_action(self, action, player):
        new_grid = self.grid.copy()
        new_grid[action] = player
        new_state = State(new_grid, action)
        # new_state.occupied_corners = self.occupied_corners.copy()
        # new_state.occupied_edges = self.occupied_edges.copy()
        # new_state.update_occupied_corners_and_edges(action, player)
        update_occupied_corners_and_edges(action,player)
        update_bonus_grid(action,player)
        perform_union(new_state.grid,action)
        return new_state
    
    def take_action_without_uf(self, action, player):
        new_grid = self.grid.copy()
        new_grid[action] = player
        new_state = State(new_grid, action)
        # new_state.occupied_corners = self.occupied_corners.copy()
        # new_state.occupied_edges = self.occupied_edges.copy()
        # new_state.update_occupied_corners_and_edges(action, player)
        update_occupied_corners_and_edges(action,player)
        update_bonus_grid(action,player)
        return new_state
    

    def get_num_actions(self):
        return len(self.get_legal_actions())

    def check_winner(self,action,player):
        win = check_win_(self.grid, action, player)
        return win

    def check_winner_without_uf(self,action,player):
        win,_ = check_win(self.grid, action, player)
        return win

    def is_full(self):
        return not np.any(self.grid == 0)

class AIPlayer:

    def __init__(self, player_number: int, timer):
        global initial_player
        initial_player = player_number
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        
        precompute_all_distances()

    def get_move(self, state: np.array) -> tuple:
        global bonus_grid
        global opp_bonus_grid
        global occupied_edges,occupied_corners
        global global_state
        global neighbor_grid
        global opponent_move
        global moves_done
        global uf

        our_move=(-1,-1)
        
        for i in range(11):
            for j in range(11):
                if global_state[i,j]!=state[i,j] and state[i,j]==3-initial_player:
                    opponent_move=(i,j)
                if global_state[i,j]!=state[i,j] and state[i,j]==initial_player:
                    our_move=(i,j)

        global_state=state.copy()
        
        if opponent_move!=(-1,-1):
            perform_union(state,opponent_move)
        
        for i in range(11):
            for j in range(11):
                neighbor_grid[i, j] = {}
        precompute_neighbors()

        corners = [(0, 0), (0, 10), (5, 10), (10, 5), (0, 5), (5, 0)]
        #print(bonus_grid)
        #print(opp_bonus_grid)
        current_state = State(state)
        for idx in range(6):
            i,j= corners[idx]
            if current_state.grid[i,j]!=0:
                occupied_corners.add(idx)
        uf_copy = copy.deepcopy(uf)
        bonus_grid_copy = bonus_grid.copy()
        opp_bonus_grid_copy = opp_bonus_grid.copy()
        occupied_edges_copy = occupied_edges.copy()
        occupied_corners_copy = occupied_corners.copy()
        count = 0
        for i in range(current_state.grid.shape[0]):
            for j in range(current_state.grid.shape[1]):
                if current_state.grid[i, j] == initial_player:
                    count+=1

        if count == 0 and initial_player==1:
            corners = get_all_corners(current_state.grid.shape[0])
            for corner in corners:
                i, j = corner
                if current_state.grid[i,j] == 0:
                    global_state[i,j] = initial_player
                    for i in range(11):
                        for j in range(11):
                            neighbor_grid[i, j] = {}
                    precompute_neighbors()
                    update_bonus_grid(corner,initial_player)
                    update_occupied_corners_and_edges(corner,initial_player)
                    global_state[corner]=initial_player
                    perform_union(global_state,corner)
                    
                    # print(bonus_grid)
                    # print("before MCTS")
                    moves_done+=1
                    return corner
                
        # Step 1: Check if AI can win immediately
        legal_actions = current_state.get_legal_actions()
        for action in legal_actions:
            simulated_state = current_state.take_action_without_uf(action, self.player_number)
            if simulated_state.is_terminal_without_uf(action, self.player_number):
                #print(action,self.player_number)
                print("immediate win ",action)
                #print("winning... ,vals : ")
                #debug_()
                moves_done+=1
                return action  # Play the winning move immediately

        # Step 2: Check if opponent can win next move and block it
        opponent = 3 - self.player_number
        for action in legal_actions:
            simulated_state = current_state.take_action_without_uf(action, opponent)
            if simulated_state.is_terminal_without_uf(action, opponent):
                print("forced block ",action)
                moves_done+=1
                global_state[action]=initial_player
                perform_union(global_state,action)
                return action  # Block opponent's winning move
            
        # Step 3: Check for "Mate in 2" for AI
        mate_in_2_move = self.find_mate_in_2(self.player_number, current_state)
        if mate_in_2_move is not None:
            #debug_()
            print("mate in 2 ",mate_in_2_move)
            moves_done+=1
            #print(global_state)
            global_state[mate_in_2_move]=initial_player
            #print(global_state)
            #debug_()
            perform_union(global_state,mate_in_2_move)

            #print(mate_in_2_move,self.player_number)
            print("mate in 2 ",mate_in_2_move)
            print("winning... ,vals : ")
            #debug_()

            return mate_in_2_move

        # # Step 4: Prevent "Mate in 2" for the opponent
        # prevent_mate_in_2_move = self.prevent_mate_in_2(current_state)
        # if prevent_mate_in_2_move is not None:
        #     print("prevent mate in 2 ",prevent_mate_in_2_move)
        #     return prevent_mate_in_2_move

        if opponent_move!=(-1,-1):
            my_move= self.maintain_vc(opponent_move,state)
            if my_move!=(-1,-1):
                moves_done+=1
                global_state[my_move]=initial_player
                perform_union(global_state,my_move)
                return my_move

        bonus_grid=bonus_grid_copy.copy()
        opp_bonus_grid = opp_bonus_grid_copy.copy()
        uf=copy.deepcopy(uf_copy)
        occupied_corners=occupied_corners_copy.copy()
        occupied_edges=occupied_edges_copy.copy()
        neighbor_grid = np.empty((11, 11), dtype=object)
        for i in range(11):
            for j in range(11):
                neighbor_grid[i, j] = {}
        precompute_neighbors()
        # Step 5: Run MCTS if no immediate win, threat, or mate in 2
        # print("just before MCTS")
        # print(bonus_grid)
        print(global_state)
        mcts = MCTS(current_state)
        best_state = mcts.search(current_state, self.player_number)
        global_state=best_state.grid.copy()
        # print("exp_time ai14: ",exp_time)
        # Find the difference between the original state and the resulting state to get the move
        diff = np.argwhere(state != best_state.grid)
        if len(diff) == 1:
            print("mcts move ",tuple(diff[0]))
            moves_done+=1
            my_move=tuple(diff[0])
            print(global_state)
            global_state[my_move]=initial_player
            print(global_state,my_move)
            debug_()
            # perform_union(global_state,my_move)
            debug_()
            return tuple(diff[0])
        else:
            raise ValueError("No valid move found")

    def in_grid(self,i,j,dim):
        if i<0:
            return False
        if j>10:
            return False
        if i+j>15:
            return False
        if i-j>5:
            return False
        if j<0:
            return False 
        return True       

    def my_get_neighbours(self,dim,vertex):
        i, j = vertex
        neighbours = []        
        neighbours.append((i - 1, j))
        neighbours.append((i - 1, j + 1))
        neighbours.append((i, j + 1))
        neighbours.append((i + 1, j + 1))
        neighbours.append((i + 1, j))
        neighbours.append((i + 1, j - 1))
        neighbours.append((i, j - 1))
        neighbours.append((i - 1, j - 1))
        
        pure_neighbors=get_neighbours(dim,vertex)
        final_neighbours=[]
        for (i,j) in neighbours:
            if self.in_grid(i,j,dim)==False:
                final_neighbours.append((10,10))
            elif (i,j) in pure_neighbors:
                final_neighbours.append((i,j))
        return final_neighbours
    
    def check_neighbour_cycle(self,action,current_state):
        our_cnt=0
        their_cnt=0
        neighbours=get_neighbours(11,action)
        for (i,j) in neighbours:
            if current_state[i,j]==initial_player:
                our_cnt+=1
            elif current_state[i,j]==3-initial_player:
                their_cnt+=1
        if our_cnt==5 and their_cnt==1:
            return True
        return False
    

#### Dont try to block to VC


#### Correct vc

    def maintain_vc(self,last_action,current_state):
        neighbors = self.my_get_neighbours(11,last_action)
        neighbors = neighbors + neighbors
        print("inside maintain vc")
        for i in range(len(neighbors)//2):
            first,second,third=current_state[neighbors[i]],current_state[neighbors[i+1]],current_state[neighbors[i+2]]
            if first==initial_player and second==0 and third==initial_player:
                print("a")
                if not (self.check_neighbour_cycle(neighbors[i+1],current_state)):
                    print("b")
                    return neighbors[i+1]
            if first==3 and second==0 and third==initial_player:
                print("c")
                return neighbors[i+1]
            if first==initial_player and second==0 and third==3:
                print("d")
                return neighbors[i+1]
        return (-1,-1)
            

    def find_mate_in_2(self, player_to_move, current_state: State) -> tuple:
        """ Find a move that guarantees a win in 2 moves (Mate in 2). """
        legal_actions = current_state.get_legal_actions()
        for action in legal_actions:
            # Simulate the AI's move
            new_state = current_state.take_action_without_uf(action, player_to_move)

            if new_state.is_terminal_without_uf(action, player_to_move):
                continue  # This move already wins, handled in step 1

            # Check all opponent's possible responses
            opponent = 3 - player_to_move
            opponent_legal_actions = new_state.get_legal_actions()

            guaranteed_win = True
            for opponent_action in opponent_legal_actions:
                # Simulate the opponent's move
                opponent_state = new_state.take_action_without_uf(opponent_action, opponent)

                # Now, check if AI can win after the opponent's move
                ai_next_actions = opponent_state.get_legal_actions()
                ai_wins_next = False
                for ai_action in ai_next_actions:
                    next_state = opponent_state.take_action_without_uf(ai_action, player_to_move)
                    if next_state.is_terminal_without_uf(ai_action, player_to_move):
                        ai_wins_next = True
                        break
                # If the opponent has at least one move that avoids AI's win, this is not a mate in 2
                if not ai_wins_next:
                    guaranteed_win = False
                    break
            if guaranteed_win:
                return action  # This is a mate in 2 move
        return None  # No mate in 2 found

    def prevent_mate_in_2(self, current_state: State) -> tuple:
        """ Check if opponent has a mate in 2 and block it if possible. """
        legal_actions = current_state.get_legal_actions()

        opponent = 3 - self.player_number

        for action in legal_actions:
            # Simulate AI's move
            new_state = current_state.take_action_without_uf(action, self.player_number)
            if new_state.is_terminal_without_uf(action,self.player_number):
                continue
            final_action = self.find_mate_in_2(opponent,new_state)
            # Check if opponent has a mate in 2 from this state
            if final_action is not None:
                # If opponent can mate in 2 after this move, block this action
                return final_action  # This move prevents the opponent's mate in 2

        return None  # No mate in 2 to block
