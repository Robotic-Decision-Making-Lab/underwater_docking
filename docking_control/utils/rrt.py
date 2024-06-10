import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import time
from tqdm import tqdm
from copy import deepcopy


class RRT:

    class Node:
        """The class for defining a node in the RRT tree"""

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None

    def __init__(self):
        """Initialize the RRT class"""
        # self.filename = "/Users/rakeshvivekanandan/workspace/bluerov2_dock/data/occ_map_binary.npy"
        self.filename = "/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/data/occ_map_binary.npy"
        self.load_map()
        self.bounds = np.array([[0, self.occ_map.shape[0]], [0, self.occ_map.shape[1]]])
        # self.min_rand = 0
        # self.max_rand = 1000
        self.step_size = 0.5
        self.goal_sample_rate = 0.3
        self.max_nodes = 100000
        self.node_list = []

    def load_map(self):
        """Load the occupancy map"""
        self.occ_map = np.load(self.filename)
        # print(self.occ_map.shape)
        # fig = plt.figure()
        # plt.imshow(self.occ_map.T, cmap="binary")
        # plt.show()

    def check_hit(self, start, goal):
        """
        Returns True if there are any occupied states between start and goal
        """
        x, y = start.x, start.y
        gx, gy = goal.x, goal.y
        dx = gx - x
        dy = gy - y

        # Went off the maze
        if (
            (x < self.bounds[0][0])
            or (y < self.bounds[1][0])
            or (x >= self.bounds[0][1])
            or (y >= self.bounds[1][1])
        ):
            return True

        # Starting in an obstacle
        if self.occ_map[int(round(start.x)), int(round(start.y))] == 1.0:
            return True

        if dx == 0.0 and dy == 0.0:  # we don't actually move, so we are done
            return False

        # discretize movement into steps
        norm = max(abs(dx), abs(dy))
        dx = dx / norm
        dy = dy / norm

        # advance the robot one step at a time
        for i in range(int(norm)):
            x += dx
            y += dy
            # Went off the maze
            if (
                (x < self.bounds[0][0])
                or (y < self.bounds[1][0])
                or (x >= self.bounds[0][1])
                or (y >= self.bounds[1][1])
            ):
                return True
            # Went into collision cell
            if self.occ_map[int(x), int(y)] == 1.0:
                return True
        return False

    def planning(self):
        """The main function for planning the path

        Returns:
            path: The path from start to goal
        """
        self.node_list = [self.start]
        start_time = time.time()
        while len(self.node_list) <= self.max_nodes:
            rand_node = self.get_random_node()
            nearest_idx = self.get_nearest_node_index(rand_node)
            nearest_node = self.node_list[nearest_idx]

            # curr_node = self.node_list[-1]
            new_node = self.steer(nearest_node, rand_node)

            collision_check = self.check_hit(nearest_node, new_node)

            if not collision_check:
                self.node_list.append(new_node)

            if (new_node.x, new_node.y) == (self.goal.x, self.goal.y):
                print("Runtime: ", (time.time() - start_time))
                return self.generate_path()
            elif (
                self.compute_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y)
                <= self.step_size
            ):
                final_node = self.steer(self.node_list[-1], self.goal)
                self.node_list.append(final_node)
                print("Runtime: ", (time.time() - start_time))
                return self.generate_path()
            else:
                # print((self.node_list[-1].x, self.node_list[-1].y))
                continue

        return None

    def steer(self, from_node, to_node):
        """Steer the node from the nearest node to the random node

        Args:
            from_node: Start node
            to_node: Goal node

        Returns:
            new_node: The new node after steering
        """
        new_node = self.Node(from_node.x, from_node.y)
        extend_length = self.step_size

        vec1 = np.array([from_node.x, from_node.y])
        vec2 = np.array([to_node.x, to_node.y])
        dist = np.linalg.norm(vec2 - vec1)
        unit_vec = (vec2 - vec1) / dist

        if extend_length > dist:
            extend_length = dist

        res_vec = vec1 + (unit_vec * extend_length)
        new_node.x = res_vec[0]
        new_node.y = res_vec[1]
        new_node.parent = from_node

        return new_node

    def generate_path(self):
        """Generate the path from the start to the goal

        Returns:
            path: The path from the start to the goal
        """
        path = []
        node = self.node_list[len(self.node_list) - 1]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path[::-1]

    def compute_dist_to_goal(self, x, y):
        """Compute the Euclidean distance to the goal

        Args:
            x: x-coordinate
            y: y-coordinate

        Returns:
            dist: The Euclidean distance to the goal
        """
        dx = x - self.goal.x
        dy = y - self.goal.y
        return math.sqrt(dx**2 + dy**2)

    def get_random_node(self):
        """Get a random node in the map

        Returns:
            rand: The random node
        """
        if np.random.rand() > self.goal_sample_rate:
            rand = self.Node(
                random.uniform(self.bounds[0][0], self.bounds[0][1]),
                random.uniform(self.bounds[1][0], self.bounds[1][1]),
            )
        else:
            rand = self.Node(self.goal.x, self.goal.y)
        return rand

    def get_nearest_node_index(self, rand_node):
        """Get the index of the nearest node to the random node

        Args:
            rand_node: The random node

        Returns:
            idx: The index of the nearest node
        """
        nodes = [[node.x, node.y] for node in self.node_list]
        tree = cKDTree(nodes)
        _, idx = tree.query([rand_node.x, rand_node.y], k=1)
        return idx

    def execute(self, waypoints):
        """Execute the RRT algorithm

        Args:
            waypoints: The waypoints to be traversed
        """
        paths = []
        fig = plt.figure()
        for i in range(waypoints.shape[0] - 1):
            self.start = self.Node(waypoints[i][0], waypoints[i][1])
            self.goal = self.Node(waypoints[i + 1][0], waypoints[i + 1][1])
            self.canvas_map = deepcopy(self.occ_map)
            goal_check = self.check_hit(self.goal, self.goal)
            if goal_check:
                print("Goal state in collision!")
            else:
                path = self.planning()
                if path is not None:
                    # print("Path: ", path)
                    path_length = 0

                    # for p in path:
                    #     self.canvas_map[p[0]][p[1]] = 50

                    for u in range(1, len(path) - 1):
                        path_length += math.sqrt(
                            (path[u][0] - path[u - 1][0]) ** 2
                            + (path[u][1] - path[u - 1][1]) ** 2
                        )
                    path = np.array(path)
                    paths.append(path)

                    print("Path length: ", round(path_length, 3))
                    print("Nodes expanded: ", len(self.node_list))

                    plt.plot(path[:, 1], path[:, 0])
                    plt.scatter(path[0, 1], path[0, 0], marker="x", c="r")
                    plt.scatter(path[-1, 1], path[-1, 0], marker="x", c="r")
        plt.imshow(self.occ_map, cmap="binary")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    obj = RRT()

    waypoints = np.array([[700, 800], [1000, 700], [900, 500], [600, 500]])
    # obj.initialize_states(waypoints)
    obj.execute(waypoints)
