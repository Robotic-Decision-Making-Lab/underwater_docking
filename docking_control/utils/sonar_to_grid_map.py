import math
import numpy as np
from tqdm import tqdm

EXTEND_AREA = 1.0


def file_read(f):
    """
    Reading LIDAR laser beams (angles and corresponding distance data)
    """
    with open(f) as data:
        measures = [line.split(",") for line in data]
    angles = []
    distances = []
    for measure in measures:
        angles.append(float(measure[0]))
        distances.append(float(measure[1]))
    angles = np.array(angles)
    distances = np.array(distances)
    return angles, distances


def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points


def calc_grid_map_config(ox, oy, xy_resolution):
    """
    Calculates the size, and the maximum distances according to the the
    measurement center
    """
    min_x = round(min(ox) - EXTEND_AREA / 2.0)
    min_y = round(min(oy) - EXTEND_AREA / 2.0)
    max_x = round(max(ox) + EXTEND_AREA / 2.0)
    max_y = round(max(oy) + EXTEND_AREA / 2.0)
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    print("The grid map is ", xw, "x", yw, ".")
    return min_x, min_y, max_x, max_y, xw, yw


def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle


def generate_ray_casting_grid_map(ox, oy, xy_resolution):
    """
    The breshen boolean tells if it's computed with bresenham ray casting
    (True) or with flood fill (False)
    """
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(ox, oy, xy_resolution)
    # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]

    occupancy_map = np.zeros((x_w, y_w))
    # occupancy_map = np.ones((x_w, y_w)) / 2

    int(np.floor(-min_x / xy_resolution))  # center x coordinate of the grid map
    int(np.floor(-min_y / xy_resolution))  # center y coordinate of the grid map

    # occupancy grid computed with bresenham ray casting
    for x, y in tqdm(zip(ox, oy), total=len(ox)):
        # x coordinate of the the occupied area
        ix = int(np.floor((x - min_x) / xy_resolution))
        # y coordinate of the the occupied area
        iy = int(np.floor((y - min_y) / xy_resolution))

        # laser_beams = bresenham((center_x, center_y), (ix, iy))
        # for laser_beam in laser_beams:
        #     if (
        #         laser_beam[0] < occupancy_map.shape[0]
        #         and laser_beam[1] < occupancy_map.shape[1]
        #     ):
        #         if occupancy_map[laser_beam[0]][laser_beam[1]] != 1.0:
        #             occupancy_map[laser_beam[0]][laser_beam[1]] = 0.0

        if ix < max_x:
            occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
        if iy < max_y:
            occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
        if ix < max_x and iy < max_y:
            occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area

        occupancy_map[ix][iy] = 1.0  # occupied area 1.0

    return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution
