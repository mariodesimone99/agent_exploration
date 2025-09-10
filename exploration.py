import pygame
import numpy as np
from numpy.random import random
import os
import imageio
import matplotlib.pyplot as plt


METHOD = 'frontier'
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 255)
STATES = {'free': 0, 'occupied': 1, 'unknown': 2, 'fov': 3, 'frontier': 4, 'agent': -1}
SIGN_MOVES = {'left': -1, 'right': 1, 'up': -1, 'down': 1}
AXIS_MOVES = {'left': 0, 'right': 0, 'up': 1, 'down': 1}
OPPOSITE_MOVES = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}
COLORS = {0: WHITE, 1: RED, 2: BLACK, 3: BLUE, 4: PURPLE, -1: GREEN}

WINDOW_HEIGHT = 300
WINDOW_WIDTH = WINDOW_HEIGHT
BLOCK_SIZE = 20
EXIT_THRESHOLD = 0.9
KNOWN_THRESHOLD = 0
OCCUPIED_THRESHOLD = 0.1
RANDOM_MOVE = 0.1
ALPHA = 1
FOV = 3

def dijkstra_path(vertices, neighbors, start_node, end_node):
    unvisited = {v: float('inf') for v in vertices}
    unvisited[start_node] = 0
    previous = {v: None for v in vertices}

    while unvisited:
        current = min(unvisited, key=unvisited.get)
        if current == end_node:
            break
        for neighbor in neighbors.get(current, []):
            if neighbor in unvisited:
                alt = unvisited[current] + 1  # Assuming all edges have weight 1
                if alt < unvisited[neighbor]:
                    unvisited[neighbor] = alt
                    previous[neighbor] = current
        unvisited.pop(current)

    path = []
    while previous[end_node] is not None:
        path.append(end_node)
        end_node = previous[end_node]
    path.append(start_node)
    path = path[::-1]
    return path, len(path) - 1 # Length -1 to account for starting position

def matrix_graph(map, start_pos, end_pos):
    binary_matrix = np.array([[1 if map[i][j] != STATES['occupied'] and map[i][j] != STATES['unknown'] else 0 for j in range(len(map[0]))] for i in range(len(map))])
    binary_matrix[end_pos[0]][end_pos[1]] = 1

    vertices = set()
    neighbors = {}
    for i in range(len(binary_matrix)):
        for j in range(len(binary_matrix[0])):
            if binary_matrix[i][j] == 1:
                label = i * len(binary_matrix[0]) + j
                vertices.add(label)
                neighbors[label] = set()
                if i > 0 and binary_matrix[i-1][j] == 1:  # Up
                    neighbors[label].add((i-1) * len(binary_matrix[0]) + j)
                if i < len(binary_matrix) - 1 and binary_matrix[i+1][j] == 1:  # Down
                    neighbors[label].add((i+1) * len(binary_matrix[0]) + j)
                if j > 0 and binary_matrix[i][j-1] == 1:  # Left
                    neighbors[label].add(i * len(binary_matrix[0]) + (j-1))
                if j < len(binary_matrix[0]) - 1 and binary_matrix[i][j+1] == 1:  # Right
                    neighbors[label].add(i * len(binary_matrix[0]) + (j+1))
    start_node = start_pos[0] * len(binary_matrix[0]) + start_pos[1]
    end_node = end_pos[0] * len(binary_matrix[0]) + end_pos[1]
    return vertices, neighbors, start_node, end_node

class Map:
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, block_size=BLOCK_SIZE, colors=COLORS):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.grid_vect = np.zeros((height // block_size, width // block_size), dtype=int)
        self.colors = colors

    def draw(self, screen, path=None):
        for x in range(0, self.height, self.block_size):
            for y in range(0, self.width, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                state = self.grid_vect[x // self.block_size][y // self.block_size]
                color = self.colors[state]
                pygame.draw.rect(screen, color, rect)
        pygame.display.update()
        if path is not None:
            pygame.image.save(screen, path)
        # pygame.time.delay(200)
        
class MapHandler:
    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, block_size=BLOCK_SIZE, colors=COLORS, occ_threshold=OCCUPIED_THRESHOLD, known_threshold=KNOWN_THRESHOLD, start_pos=((WINDOW_HEIGHT // BLOCK_SIZE) // 2, (WINDOW_WIDTH // BLOCK_SIZE) // 2)):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.grid_map = Map(width, height, block_size, colors=colors)
        self.agent_map = Map(width, height, block_size, colors=colors)
        self.image_counter = 0
        self.epoch = [0]
        self.visited_cells = [1]
        self.occ_threshold = occ_threshold
        self.known_threshold = known_threshold
        self.start_pos = start_pos
        self.init_maps()

    def init_maps(self):
        for x in range(0, self.height, self.block_size):
            for y in range(0, self.width, self.block_size):

                prob_occ = random() < self.occ_threshold
                prob_unk = random() < self.known_threshold
                if prob_occ:    
                    self.grid_map.grid_vect[x // self.block_size][y // self.block_size] = STATES['occupied']
                if prob_unk:
                    self.agent_map.grid_vect[x // self.block_size][y // self.block_size] = self.grid_map.grid_vect[x // self.block_size][y // self.block_size]
                else:
                    self.agent_map.grid_vect[x // self.block_size][y // self.block_size] = STATES['unknown']
        # Add Agent
        self.agent_map.grid_vect[self.start_pos[0]][self.start_pos[1]] = STATES['agent']
        self.grid_map.grid_vect[self.start_pos[0]][self.start_pos[1]] = STATES['free']

    def draw_save(self, screen, method, save_path=None):
        if save_path is not None:
            self.agent_map.draw(screen, path=os.path.join(save_path, f"{method}_{self.image_counter}.png"))
            self.image_counter += 1
        else:
            self.agent_map.draw(screen)

    def greedy_update(self, screen, fov=FOV, alpha=ALPHA, prob_random=RANDOM_MOVE, last_move=None, path=None):
        dir_move, agent_idx = self.find_direction()
        dir = self.greedy_fov(dir_move, agent_idx, screen, fov=fov, path=path)
        legal_dir = self.find_legal(dir)
        if random() < prob_random:
            new_move = {}
            random_move = self.random_move(legal_dir)
            random_dir = list(random_move.values())[0]
            random_dir = np.random.randint(1, random_dir + 1) if random_dir > 0 else np.random.randint(random_dir, 0)
            new_move[list(random_move.keys())[0]] = random_dir
        else:
            new_move = self.find_opt(legal_dir, alpha)
            new_dir = list(new_move.keys())[0]
            if self.check_loop(last_move, new_dir):
                if len(legal_dir['move']) > 1:
                    del legal_dir['move'][new_dir]
                    del legal_dir['cost'][new_dir]
                    del legal_dir['gain'][new_dir]
                else:
                    dir_move, agent_idx = self.find_direction()
                    dir = self.greedy_fov(dir_move, agent_idx, screen, fov=1, path=path)
                    legal_dir = self.find_legal(dir)

            new_move = self.find_opt(legal_dir, alpha)
        new_cells = 0
        for d in dir['move']:
            new_cells += dir['gain'][d]
        self.greedy_move(agent_idx, new_move, screen, path=path)
        return new_move
    
    def find_direction(self, pos=None):
        if pos is None:
            agent_idx = np.where(self.agent_map.grid_vect == STATES['agent'])
            agent_idx = (agent_idx[0][0], agent_idx[1][0])
        else:
            agent_idx = pos

        dir_move = {}
        if agent_idx[1] > 0:
            dir_move['left'] = -1
        if agent_idx[1] < len(self.agent_map.grid_vect[0]) - 1:
            dir_move['right'] = 1
        if agent_idx[0] > 0:
            dir_move['up'] = -1
        if agent_idx[0] < len(self.agent_map.grid_vect) - 1:
            dir_move['down'] = 1
        return dir_move, agent_idx

    def greedy_fov(self, directions, agent_idx, screen, fov, path=None):
        cost = {d: 0 for d in directions}
        gain = {d: 0 for d in directions}
        new_cells = 0
        for d in directions:
            axis_fix = AXIS_MOVES[d]
            sign = SIGN_MOVES[d]
            axis_mov = 0 if axis_fix == 1 else 1
            mov_range = range(agent_idx[axis_mov] + 1, agent_idx[axis_mov] + fov + 1) if sign > 0 else range(agent_idx[axis_mov] - 1, agent_idx[axis_mov] - fov - 1, -1)
            stop_iter = False

            p = 1
            for i in mov_range:
                a_map = self.agent_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.agent_map.grid_vect[agent_idx[axis_fix], :]
                g_map = self.grid_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.grid_map.grid_vect[agent_idx[axis_fix], :]
                if a_map[i] == STATES['unknown']:
                    gain[d] += 1
                    new_cells += 1
                a_map[i] = STATES['fov']
                self.draw_save(screen, METHOD, save_path=path)
                # Suppose to not be able to view beyond objects
                # Check if the border of the map is reached or an object is encountered
                if g_map[i] == STATES['occupied']:
                    stop_iter = True
                    directions[d] = p-1 if sign > 0 else -p+1
                elif (i == 0 and d in ['left', 'up']) or (i == len(g_map) - 1 and d in ['right', 'down']):
                    stop_iter = True
                    directions[d] = p if sign > 0 else -p
                else:
                    directions[d] = p if sign > 0 else -p
                
                cost[d] += 1
                p += 1
                if stop_iter:
                    break
            start_point = agent_idx[axis_mov] + 1 if sign > 0 else agent_idx[axis_mov] - p +1
            end_point = start_point + p - 1 if sign > 0 else agent_idx[axis_mov]

            a_elements = self.agent_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.agent_map.grid_vect[agent_idx[axis_fix], :]
            g_elements = self.grid_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.grid_map.grid_vect[agent_idx[axis_fix], :]
            a_elements[start_point:end_point] = g_elements[start_point:end_point]
        self.visited_cells.append(self.visited_cells[-1] + new_cells)
        self.epoch.append(self.epoch[-1] + 1)
        return {'move': directions, 'cost': cost, 'gain': gain}

    def greedy_reward(self, directions, alpha):
        reward = {i: 0 for i in directions['move'].keys()}
        for d in directions['move']:
            reward[d] = directions['gain'][d] - alpha*directions['cost'][d]
        return reward
    
    def check_loop(self, last_move, new_dir):
        if len(last_move) < 2:
            return False
        elif last_move[0] == new_dir and last_move[1] == OPPOSITE_MOVES[new_dir]:
            return True
        return False
    
    def find_opt(self, dir, alpha):
        reward = self.greedy_reward(dir, alpha)
        opt_reward = np.max(list(reward.values()))
        moves = []
        for d in dir['move']:
            if reward[d] == opt_reward:
                moves.append(d)
        opt_mov = np.random.randint(0, len(moves))
        new_dir = moves[opt_mov]
        new_move = {new_dir: dir['move'][new_dir]}
        return new_move
    
    def random_move(self, legal_dir):
        opt_mov = np.random.choice(list(legal_dir['move'].keys()))
        new_dir = legal_dir['move'][opt_mov]
        new_move = {opt_mov: new_dir}
        return new_move

    def find_legal(self, dir):
        legal_dir = {'move': {}, 'cost': {}, 'gain': {}}
        for d in dir['move']:
            if dir['move'][d] != 0:
                legal_dir['move'][d] = dir['move'][d]
                legal_dir['cost'][d] = dir['cost'][d]
                legal_dir['gain'][d] = dir['gain'][d]
        return legal_dir
    
    def greedy_move(self, agent_idx, direction, screen, path=None):
        d_move = list(direction.keys())[0]
        axis_fix = AXIS_MOVES[d_move]
        axis_mov = 0 if axis_fix == 1 else 1
        sign = SIGN_MOVES[d_move]
        mov_range = range(agent_idx[axis_mov] + 1, agent_idx[axis_mov] + direction[d_move] + 1) if sign > 0 else range(agent_idx[axis_mov] - 1, agent_idx[axis_mov] - np.abs(direction[d_move]) - 1, -1)
        for i in mov_range:
            a_map = self.agent_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.agent_map.grid_vect[agent_idx[axis_fix], :]
            g_map = self.grid_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.grid_map.grid_vect[agent_idx[axis_fix], :]
            a_map[i] = STATES['agent']
            if sign > 0:
                a_map[i-1] = g_map[i-1]
            else:
                a_map[i+1] = g_map[i+1]
            self.draw_save(screen, METHOD, save_path=path)

    def find_frontier(self):
        idx_list = []
        # Find frontier
        for i in range(len(self.agent_map.grid_vect)):
            for j in range(len(self.agent_map.grid_vect[0])):
                if self.agent_map.grid_vect[i][j] != STATES['unknown']:
                    continue
                test_frontier = False
                if i > 0 and self.agent_map.grid_vect[i-1][j] == STATES['free']:  # Up
                    test_frontier = True

                if i < len(self.agent_map.grid_vect) - 1 and self.agent_map.grid_vect[i+1][j] == STATES['free']:  # Down
                    test_frontier = True

                if j > 0 and self.agent_map.grid_vect[i][j-1] == STATES['free']:  # Left
                    test_frontier = True

                if j < len(self.agent_map.grid_vect[0]) - 1 and self.agent_map.grid_vect[i][j+1] == STATES['free']:  # Right
                    test_frontier = True
                if test_frontier:
                    idx_list.append((i, j))

        return idx_list

    def frontier_gain(self, idx, fov):
    # Compute potential gain for a frontier cell, it is not the real gain because it does not account for the cost of moving and obstacles
        fov_dir, _ = self.find_direction(idx)
        gain = 0
        for d in fov_dir:
            axis_fix = AXIS_MOVES[d]
            sign = SIGN_MOVES[d]
            axis_mov = 0 if axis_fix == 1 else 1
            mov_range = range(idx[axis_mov], idx[axis_mov] + fov + 1) if sign > 0 else range(idx[axis_mov], idx[axis_mov] - fov - 1, -1)

            for i in mov_range:
                a_map = self.agent_map.grid_vect[:, idx[axis_fix]] if axis_fix == 1 else self.agent_map.grid_vect[idx[axis_fix], :]
                if a_map[i] == STATES['unknown']:
                    gain += 1
                if (i == 0 and d in ['left', 'up']) or (i == len(a_map) - 1 and d in ['right', 'down']):
                    break
        return gain
    
    def frontier_cost(self, goal):
        agent_idx = np.where(self.agent_map.grid_vect == STATES['agent'])
        agent_idx = (agent_idx[0][0], agent_idx[1][0])

        vertices, neighbors, start_node, end_node = matrix_graph(self.agent_map.grid_vect, agent_idx, goal)
        path, cost = dijkstra_path(vertices, neighbors, start_node, end_node)
        return path, cost
    
    def frontier_path(self, screen, fov, alpha, save_path=None):
        frontier_list = self.find_frontier()
        opt_path = []
        opt_reward = np.NINF
        opt_idx = None
        for idx in frontier_list:
            self.agent_map.grid_vect[idx[0]][idx[1]] = STATES['frontier']
            self.draw_save(screen, METHOD, save_path=save_path)
            
            idx_gain = self.frontier_gain(idx, fov)
            idx_path, idx_cost = self.frontier_cost(idx)
            idx_reward = idx_gain - alpha * idx_cost
            if idx_reward > opt_reward:
                opt_reward = idx_reward
                opt_path = idx_path
                opt_idx = idx
        for idx in frontier_list:
            if idx != opt_idx:
                self.agent_map.grid_vect[idx[0]][idx[1]] = STATES['unknown']
        return opt_path

    def frontier_move(self, screen, opt_path, save_path=None):
        ncols = self.agent_map.grid_vect.shape[1]
        for node in opt_path[1:-1]:
            i = node // ncols
            j = node % ncols
            # Save previous indexes to avoid searching for the agent at each step
            agent_idx = np.where(self.agent_map.grid_vect == STATES['agent'])
            agent_idx = (agent_idx[0][0], agent_idx[1][0])
    
            self.agent_map.grid_vect[i][j] = STATES['agent']
            self.agent_map.grid_vect[agent_idx[0]][agent_idx[1]] = STATES['free']
            self.draw_save(screen, METHOD, save_path=save_path)
        i_frontier = opt_path[-1] // ncols
        j_frontier = opt_path[-1] % ncols
        self.agent_map.grid_vect[i_frontier][j_frontier] = STATES['fov']
        self.draw_save(screen, METHOD, save_path=save_path)
        self.visited_cells.append(self.visited_cells[-1] + 1)
        if self.grid_map.grid_vect[i_frontier][j_frontier] == STATES['occupied']:
            self.agent_map.grid_vect[i_frontier][j_frontier] = STATES['occupied']
        else:
            self.agent_map.grid_vect[i][j] = STATES['free']
            self.agent_map.grid_vect[i_frontier][j_frontier] = STATES['agent']
        self.draw_save(screen, METHOD, save_path=save_path)

    def frontier_fov(self, agent_idx, screen, fov, save_path=None):
        dir_move, _ = self.find_direction(agent_idx)
        gain = 0
        for d in dir_move:
            axis_fix = AXIS_MOVES[d]
            sign = SIGN_MOVES[d]
            axis_mov = 0 if axis_fix == 1 else 1
            mov_range = range(agent_idx[axis_mov] + 1, agent_idx[axis_mov] + fov + 1) if sign > 0 else range(agent_idx[axis_mov] - 1, agent_idx[axis_mov] - fov - 1, -1)
            
            p = 1
            stop_iter = False
            for i in mov_range:
                a_map = self.agent_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.agent_map.grid_vect[agent_idx[axis_fix], :]
                g_map = self.grid_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.grid_map.grid_vect[agent_idx[axis_fix], :]

                if a_map[i] == STATES['unknown']:
                    gain += 1
                a_map[i] = STATES['fov']
                self.draw_save(screen, METHOD, save_path=save_path)
                # Suppose to not be able to view beyond objects
                # Check if the border of the map is reached or an object is encountered
                if g_map[i] == STATES['occupied']:
                    stop_iter = True
                elif (i == 0 and d in ['left', 'up']) or (i == len(g_map) - 1 and d in ['right', 'down']):
                    stop_iter = True
                p += 1
                
                if stop_iter:
                    break
            start_point = agent_idx[axis_mov] + 1 if sign > 0 else agent_idx[axis_mov] - p +1
            end_point = start_point + p - 1 if sign > 0 else agent_idx[axis_mov]

            a_elements = self.agent_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.agent_map.grid_vect[agent_idx[axis_fix], :]
            g_elements = self.grid_map.grid_vect[:, agent_idx[axis_fix]] if axis_fix == 1 else self.grid_map.grid_vect[agent_idx[axis_fix], :]
            a_elements[start_point:end_point] = g_elements[start_point:end_point]
        self.visited_cells[-1] += gain
        self.epoch.append(self.epoch[-1] + 1)

    def frontier_explore(self, screen, fov=FOV, alpha=ALPHA, save_path=None):
        agent_idx = np.where(self.agent_map.grid_vect == STATES['agent'])
        agent_idx = (agent_idx[0][0], agent_idx[1][0])
        self.frontier_fov(agent_idx, screen, fov, save_path=save_path)
        opt_path = self.frontier_path(screen, fov, alpha, save_path)
        if opt_path == []:
            print("No more frontiers to explore.")
            return True
        self.frontier_move(screen, opt_path, save_path)
        return False


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Frontier Exploration")
    map_handler = MapHandler(occ_threshold=OCCUPIED_THRESHOLD, known_threshold=KNOWN_THRESHOLD)

    done = False
    last_move = []
    path_steps = 'exploration_steps'
    if not os.path.exists(path_steps):
        os.makedirs(path_steps)
    while not done:
        map_handler.agent_map.draw(screen)
        if METHOD == 'frontier':
            done = map_handler.frontier_explore(screen, fov=FOV, alpha=ALPHA, save_path=path_steps)
        else:
            if len(last_move) == 3:
                last_move.pop(0)
            new_move = map_handler.greedy_update(screen, fov=FOV, alpha=ALPHA, prob_random=RANDOM_MOVE, last_move=last_move, path=path_steps)
            last_move.append(list(new_move.keys())[0])

        known_cells = np.sum(map_handler.agent_map.grid_vect == STATES['free']) + np.sum(map_handler.agent_map.grid_vect == STATES['occupied'])
        total_cells = map_handler.agent_map.grid_vect.size - 1 # Exclude the agent cell
        if known_cells / total_cells >= EXIT_THRESHOLD:
            print("Exploration complete!")
            done = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    map_handler.agent_map.draw(screen)
    map_handler.frontier_explore(screen, fov=FOV, alpha=ALPHA)
    images = []
    print("Saving results...")
    filenames = sorted(os.listdir(path_steps), key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in filenames:
        images.append(imageio.imread(os.path.join(path_steps, filename)))
    imageio.mimsave(f'{METHOD}.gif', images)
    print("Results saved")
    for filename in filenames:
        os.remove(os.path.join(path_steps, filename))
    os.rmdir(path_steps)
    print("Temporary files removed")
    print("Saving Statistics...")
    plt.plot(map_handler.epoch, map_handler.visited_cells)
    plt.xlabel('Epoch')
    plt.ylabel('Visited Cells')
    plt.title('Number of Cells Explored vs Time')
    plt.savefig(f'{METHOD}_statistic.png')
    plt.show()
    print("Statistics saved")

main()