# -*- coding: utf-8 -*-
"""Untitled23.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PT2ftRE8OyrNegvrNDoUMu3hMraSwcOi
"""

# connecting to Google drive to save data
from google.colab import drive
import os

drive.mount('/content/drive/')
print(os.chdir('/content/drive/My Drive/research_Data'))
os.listdir(os.getcwd())

"""# Pre-generation of background grid
The finite element mesh elements divided by the background grid method are of high quality and uniform size, which is beneficial to the finite element calculation. First, it is neccessary to specify the size of the concrete sample (For 2D case,: L *X* H) and each element size is ecell. It should be noted that the element size needs to be divisible by each side of the concrete sample. The value of the element size should be reasonable.
"""

# specification of the concrete sample
# 2D (L * H)


# all units are in mm

L = 500
H = 500

# meaning each cell will be 50*50
cell = 50*50  # area of cell
concrete = 500*500  # area of concrete specimen
num_of_cells = concrete / cell  # total area / cell area = number of cells
print(num_of_cells)

def generate_vertices(j, i, ecell):
  """returns array of a, b, c, d, o(center)"""

  # vertices
  a = [(i-1)*ecell, (j-1)*ecell]
  b = [i*ecell, (j-1)*ecell]
  c = [i*ecell, j*ecell]
  d = [(i-1)*ecell, j*ecell]

  #center
  o = [(i-0.5)*ecell, (j-0.5)*ecell]

  return [a, b, c, d, o]

def generate_grid(size, num_cell=10, file_name='2D_mesh.txt'):
  """This will generate a grid"""
  L = size
  H = size
  cell = int(size / num_cell)

  Ldiv = L / cell
  Hdiv = L / cell
  Nelem = Ldiv * Hdiv

  # I will return a object with its key as
  twoD_grid = {}
  twoD = []

  # open a file
  f = open(f"{file_name}", "w")

  for i in range(num_cell):
    i += 1
    for j in range(num_cell):
      j += 1
      # a b c d
      content = twoD_grid[f'{i}, {j}'] = generate_vertices(i, j, cell)
      f.write(f"({i}, {j}): {str(content)} \n")
      twoD.append(generate_vertices(i, j, cell))

  f.close()

  return twoD_grid, twoD

"""# Generation of a Polygonal Aggregate
The parameters controlling the position and shape of the polygonal aggregates are the centroid coordinates O(xo, yo) and vertex coordinates Pi. To obtain the vertex coordinates within the particle size range (di, di+1), a series of polar
angles theta_i and polar radius radius_i are randomly generated in the polar coordinate sytem (why the polar coordinate system?).

The polygonal aggregate generated by the above method include concave polygons and convex polygons.

"""

import random
import math

def number_of_polygon_sides(n):
  """probability function for the number of polygon sides n."""

  # both the data below is obtained by fitting on the statistics
  omega = 2.2  # as given in the paper
  nc = 5.8

  return (1/(omega*math.sqrt(math.pi/2)))*math.exp((-2*(n-nc)**2)/omega**2)


def first_angle(n, dkl, dku):
  """After defining an origin inside the aggregate. set the polar
  angle theta of the first corner"""

  # random number between 0 and 1
  nl = random.random()
  angle = (nl/n)*2*math.pi
  # calculate radius
  rad = radius(dkl, dku, nl)

  return angle, rad


def other_angle(n, dkl, dku):
  """
  setting the polar angle and polar radius of the ith
  corner Pi in a similar way.
  """
  nli = random.random()

  denominator = sum([((2/n) + nli) for _ in range(2, n)])
  numerator = ( (2*math.pi/n) + (2 * math.pi * nli))
  angle = numerator / denominator

  # calculate radius
  rad = radius(dkl, dku, nli)

  return angle, rad

def radius(dkl, dku, nli):
  return ((dkl+dku)/4) + (2*nli - 1) * ((dku-dkl)/4)

def fuller_curve(d):
  m = 0.5
  # upper limit of aggregate size
  D = 9.5
  return ((d/D)**m)

# This function gets just one pair of coordinates based on the angle theta
def get_circle_coord(theta, x_center, y_center, radius):
    x = radius * math.cos(theta) + x_center
    y = radius * math.sin(theta) + y_center
    return (x,y)

import random
import math
import numpy as np
import matplotlib.pyplot as plt

# I want to draw a polygonal aggregate
def draw_polygon():
  """Thus will generate a series of polar angles
  theta and radii"""

  # aggregate size ranges (lower limit, upper limit)
  aggregate_size_range = [[2.5, 4.75], [4.75, 9.5], [9.5, 12.5], [12.5, 19.0]]
  # aggregate_size_range = [[5, 10], [10, 20], [20, 30], [30, 40], [40, 50],
  #  [50, 60], [60, 70], [70, 80]]
  # the number of vertices of a polygon, will be randomly selected
  # aggregate_size_range = [[0.15, 0.300], [0.300, 0.6], [0.6, 1.18],
  #   [1.18, 2.36], [2.36, 4.75], [4.75, 9.5]]

  # this will compute the cumulative probability that an aggregate..
  # passes a sieve opening with a size of d mm.
  probabilities = [fuller_curve(i[1]) for i in aggregate_size_range]
  probabilities = probabilities[::-1]
  print(probabilities)

  # random.choices will randomly select an aggregate size range...
  # while considering that each size range has a weight value...
  # that determines the probability of its occurance.
  size = random.choices([i[1] for i in aggregate_size_range], weights = probabilities)

  num_vertices = [6, 7, 8, 9, 10]
  # select a random vertice number
  n = random.choice(num_vertices)
  # randomly select size range
  size_range = random.choice(aggregate_size_range)
  #lower limit
  di = size_range[0]
  # upper limit
  di1 = size_range[1]

  print(size_range)


  # array of the probability of an n-sided aggregate occuring.
  probabilities = [number_of_polygon_sides(i) for i in num_vertices]
  # randomly picking the n-sides of an aggregates based on the probability function
  n = random.choices(num_vertices, probabilities)[0]
  print(f"sides: {n}")

  # calculate first angle and radius
  fst_angle, first_radius = first_angle(n, di, di1)

  # second angle and radius calculation
  angle_and_radius = []
  print(f'n: {n}, dkl: {di}, dku: {di1}')
  ith = 2
  while ith <= n:
    angle, radius = other_angle(n, di, di1)
    angle_and_radius.append([angle, radius])
    ith += 1

  angle_in_rads = []

  angle_and_radius.insert(0, (fst_angle, first_radius))

  for index, i in enumerate(angle_and_radius):
    if index == 0:
      angle_in_rads.append(i[0])
    else:
      sum_of_angle = sum([i[0] for i in angle_and_radius[:index]])
      angle_in_rads.append(i[0] + sum_of_angle)

  # print(f"Angle in degrees: {angle_in_rads}")
  # print(f"Angle and radius: {angle_and_radius}")

  return angle_and_radius, angle_in_rads


import random


def polarToCartesian(angle_and_radius, angle_in_rads, origin=None):

  x_coords = []
  y_coords = []
  polygon = []

  for index, i in enumerate(angle_and_radius):
    x_coord = i[1] * np.cos(angle_in_rads[index])
    y_coord = i[1] * np.sin(angle_in_rads[index])

    x_coords.append(x_coord)
    y_coords.append(y_coord)
    # because we have not finished working on x_coords and y_coords
    # polygon.append((x_coord, y_coord))


  if not origin:
    r1, r2 = random.randint(0, 500), random.randint(0, 500)

  origin = (r1, r2)

  # Calculate the coordinates with respect to the origin
  # x_coordinates, y_coordinates = zip(*polygon)
  origin_x, origin_y = origin
  x_coords = [x + origin_x for x in x_coords]
  y_coords = [y + origin_y for y in y_coords]
  for x_coord, y_coord in zip(x_coords, y_coords):
    polygon.append((x_coord, y_coord))


  return x_coords, y_coords, polygon, origin



def plotPolyGon(c_coords):
  x_coords, y_coords = c_coords[0], c_coords[1]
  # Create a line plot to connect the points and form a closed polygon
  plt.figure()
  plt.plot(x_coords, y_coords, marker='o')
  plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 'k-')
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title('Polygon from Set of Angles')
  plt.grid(True)
  plt.axis('equal')
  plt.show()

  # return (x_coords, y_coords)
p_coords = draw_polygon()
c_coords = polarToCartesian(p_coords[0], p_coords[1])

"""We now need to be able to plot the polygon at random locations  of  the concrete size."""

import matplotlib.pyplot as plt

def plotFilledPolyGonConcrete(c_coords, x_limit=None, y_limit=None):
    x_coords, y_coords = c_coords[0], c_coords[1]

    # Create a filled polygon
    plt.figure()
    plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Filled Polygon from Set of Angles')
    plt.grid(True)

    if x_limit is not None:
        plt.xlim(x_limit)
    if y_limit is not None:
        plt.ylim(y_limit)

    plt.show()

x_limit = (0, 500)
y_limit = (0, 500)
plotFilledPolyGonConcrete(c_coords, x_limit, y_limit)

"""We need to have a way of checking that  the x and  y coordinate vertices of a  polygon does  not overlap the x, y axis  of  the plot"""

def hasOverlapWithAxes(x_coords, y_coords):
    # Check if any x or y coordinate is zero or negative
    if any(coord <= 0 for coord in x_coords) or any(coord <= 0 for coord in y_coords) \
    or any(coord >= 500 for coord in x_coords) or any(coord >=  500 for coord in y_coords):
        return True
    else:
        return False

def plotPolyGonConcrete(c_coords, x_limit=None, y_limit=None):
    x_coords, y_coords = c_coords[0], c_coords[1]

    # Check for overlap with axes
    if hasOverlapWithAxes(x_coords, y_coords):
        print("Polygon vertices overlap with the x or y axis. Adjust coordinates.")
        # return

     # Create a filled polygon
    plt.figure()
    plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Filled Polygon from Set of Angles')
    plt.grid(True)

    if x_limit is not None:
        plt.xlim(x_limit)
    if y_limit is not None:
        plt.ylim(y_limit)

    plt.show()


x_limit = (0, 500)
y_limit = (0, 500)

# This will print a message indicating overlap with axes
plotPolyGonConcrete(c_coords, x_limit, y_limit)

"""We have to write code to ensure that the points of a polygon do  not intersect or overlap with another polygon."""

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def doPolygonsOverlap(x_coords1, y_coords1, x_coords2, y_coords2):
    poly1 = Polygon(zip(x_coords1, y_coords1))
    poly2 = Polygon(zip(x_coords2, y_coords2))

    return poly1.intersects(poly2)

# Generate the polygon coordinates using draw_polygon
p_coords1 = draw_polygon()
c_coords1 = polarToCartesian(p_coords1[0], p_coords1[1])

p_coords2 = draw_polygon()
c_coords2 = polarToCartesian(p_coords2[0], p_coords2[1])

 # Create a filled polygon
plt.figure()

for c_coords in [c_coords1, c_coords2]:

    x_coords, y_coords = c_coords[0], c_coords[1]

    # Check for overlap with axes
    if hasOverlapWithAxes(x_coords, y_coords):
        print("Polygon vertices overlap with the x or y axis. Adjust coordinates.")
        # return




    plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

# print('x_coords: ', c_coords1[0])
# print('x_coords: ', c_coords2[0])
# print('y_coords: ', c_coords1[1])
# print('y_coords: ', c_coords2[1])

if doPolygonsOverlap(c_coords1[0], c_coords1[1], c_coords2[0], c_coords2[1]):
    print("Polygons overlap.")
else:
    print("Polygons do not overlap.")


plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Filled Polygon from Set of Angles')
plt.grid(True)

if x_limit is not None:
    plt.xlim(x_limit)
if y_limit is not None:
    plt.ylim(y_limit)

plt.show()

"""We have to create a large database of aggregates"""

generated_aggregates = []
# grids = generate_grid(500)[1]
for i in range(50000):
  p_coords = draw_polygon()
  c_coords = polarToCartesian(p_coords[0], p_coords[1])
  generated_aggregates.append(c_coords)

"""We have not generated 50,000 aggregates. we will now check to ensure that the origin of the aggregates do not intersect with the axis of the concrete. We will create a new list  of aggregates that meet this requirements and pop aggregates that do not meet this requirement."""

def hasOverlapWithAxes(x_coords, y_coords):
    # Check if any x or y coordinate is zero or negative
    if any(coord <= 0 for coord in x_coords) or any(coord <= 0 for coord in y_coords) \
    or any(coord >= 500 for coord in x_coords) or any(coord >=  500 for coord in y_coords):
        return True
    else:
        return False

non_overlap_generated_aggregates = []

for aggregate in generated_aggregates:
  x_coords, y_coords = aggregate[0], aggregate[1]

  # Check for overlap with axes
  if hasOverlapWithAxes(x_coords, y_coords):
    print("Polygon vertices overlap with the x or y axis. Adjust coordinates.")
  else:
    non_overlap_generated_aggregates.append(aggregate)
    print("Does not overlap")

print(len(generated_aggregates))
print(len(non_overlap_generated_aggregates))
value = len(non_overlap_generated_aggregates)/len(generated_aggregates)*100
print(f"Percent of aggregates overlapping with the sides of the concrete removed: {round(100-value, 2)}%")

"""We also need to check for overlapping and intersection between aggregates. If any two aggregates intersect remove the one being checked on. Do  this for all aggregates"""

def doPolygonsOverlap(x_coords1, y_coords1, x_coords2, y_coords2):
    poly1 = Polygon(zip(x_coords1, y_coords1))
    poly2 = Polygon(zip(x_coords2, y_coords2))

    return poly1.intersects(poly2)

# copying a list using slicing
copy_non_overlap_aggregates = non_overlap_generated_aggregates[:]

indexes = []
for i in range(len(non_overlap_generated_aggregates[0:2000])):
  first_aggregate = non_overlap_generated_aggregates[i]
  for j in range(i+1, len(non_overlap_generated_aggregates[0:2000])):
    second_aggregate = non_overlap_generated_aggregates[j]
    #carry out the comparision
    if doPolygonsOverlap(first_aggregate[0], first_aggregate[1], second_aggregate[0], second_aggregate[1]):
      # del non_overlap_generated_aggregates[j]
      indexes.append(j)
      print("Polygon's overlap")
    else:
      print("Polygons do not overlap")

  print(f'Done with round: {i+1}')

print('end of program')

"""Remove all repeating indexes from the list of indexes"""

non_repeating_indexes =  []
for index  in  indexes:
  if index not in non_repeating_indexes:
    non_repeating_indexes.append(index)

print(len(non_repeating_indexes))

"""Delete all the aggregates that  overlap from our database of 500 aggregates"""

non_overlap_aggregates = [i for index, i in enumerate(non_overlap_generated_aggregates[:500]) if index not in non_repeating_indexes]

print(len(non_overlap_aggregates))

"""I now need to add all aggregates to the concrete."""

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

plt.figure()

for c_coords in non_overlap_aggregates:

    x_coords, y_coords = c_coords[0], c_coords[1]

    plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Filled Polygon from Set of Angles')
plt.grid(True)

if x_limit is not None:
    plt.xlim(x_limit)
if y_limit is not None:
    plt.ylim(y_limit)

plt.show()

"""Now I need to carry out material identification and seperate the aggregate from the matrix and the interfacial transition zone (ITF).

Plotting with grids
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_colored_cells():
    # Define the size of each cell
    cell_size = 5
    num_cells = 500 // cell_size

    # Create a new figure
    plt.figure()

    # Create axis
    ax = plt.gca()

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)

    # Create gridlines for the cells
    for x in range(0, 501, cell_size):
        ax.axvline(x, color='black', linestyle='-', linewidth=1)

    for y in range(0, 501, cell_size):
        ax.axhline(y, color='black', linestyle='-', linewidth=1)

    # Fill each cell with a color
    colors = np.random.rand(num_cells, num_cells, 3)  # Random colors for each cell
    for i in range(num_cells):
        for j in range(num_cells):
            # cell_color = colors[i, j, :]
            if i == 0 and j  <= 2:
              ax.fill_between([i * cell_size, (i + 1) * cell_size], [j * cell_size, j * cell_size], [(j + 1) * cell_size, (j + 1) * cell_size], color='white')
            else:
              ax.fill_between([i * cell_size, (i + 1) * cell_size], [j * cell_size, j * cell_size], [(j + 1) * cell_size, (j + 1) * cell_size], color='green')


    for c_coords in non_overlap_aggregates:
      x_coords, y_coords = c_coords[0], c_coords[1]
      plt.fill(x_coords, y_coords, facecolor='red', edgecolor='white')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Grid of Colored Cells')

    # Display the plot
    plt.grid(True)
    plt.show()

# Call the function to generate the plot
plot_colored_cells()

"""I have to create an array of 10,000 grids and check if the centroid of the grid lies within one of the polygons, if it lies inside, inside=true, outside=false, edge=false. If it lies outside, inside=false, edge=false, outside=true. If it  lies at the edge. inside=false, edge=true, outside=false."""

# dimensions of the concrete
L = 500
H = 500

# each cell  will be 5*5
cell = 5*5
concrete = L * H
num_of_cells = concrete / cell


# vertices of the cell
def generate_vertices(j, i, ecell):
  """returns array of a, b, c, d, o(center)"""

  # vertices
  a = [(i-1)*ecell, (j-1)*ecell]
  b = [i*ecell, (j-1)*ecell]
  c = [i*ecell, j*ecell]
  d = [(i-1)*ecell, j*ecell]

  #center
  o = [(i-0.5)*ecell, (j-0.5)*ecell]

  return [a, b, c, d, o]

def generate_grid(size, num_cell=100, file_name='2D_mesh.txt'):
  """This will generate a grid"""
  L = size
  H = size
  cell_area = 5*5
  concrete_area =  L*H
  num_of_cells = concrete_area / cell_area
  # num_cell = int(L / 5); (no need)
  print('num_cell', num_cell)
  cell = int(size / num_cell)

  Ldiv = L / cell  # 100
  Hdiv = L / cell  #100
  Nelem = Ldiv * Hdiv  #10,000

  # I will return a object with its key as
  twoD_grid = {}
  twoD = []

  # open a file
  f = open(f"{file_name}", "w")

  for i in range(num_cell):
    i += 1
    for j in range(num_cell):
      j += 1
      # a b c d
      content = twoD_grid[f'{i}, {j}'] = generate_vertices(i, j, cell)
      f.write(f"({i}, {j}): {str(content)} \n")
      # inside, edge, outside
      conditions = [False, False, False]
      twoD.append(generate_vertices(i, j, cell) + conditions)

  f.close()

  return twoD_grid, twoD


generated_grids = generate_grid(500)[1]
print(generated_grids)
print(len(generated_grids))

"""We have to check the relationship between the background grid and all the non_overlap polygons."""

# getting a general idea of the generated grid.
print(generated_grids[0][:4])
# getting my generated polygon
# for c_coords in non_overlap_aggregates
print(len(non_overlap_aggregates))

"""Relationship between a point (x, y) and a polygon. So that we can identify when a cell is located within an aggregate, outside or at its  boundary."""

# checking if a single point is inside a polygon.
def is_point_inside_polygon(x, y, polygon):
    n = len(polygon)
    wn = 0  # Initialize winding number

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]  # Wrap around for the last edge

        if y1 <= y:
            if y2 > y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0:
                wn += 1
        elif y2 <= y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0:
            wn -= 1

    return wn != 0  # Point is inside if winding number is not zero

# Example usage
c_coords = polarToCartesian(p_coords[0], p_coords[1])
print(c_coords[3])
print(c_coords[0])
print(c_coords[1])
print(c_coords[2])
plotPolyGon(c_coords)
polygon = c_coords[2]
point = (10,10)
is_inside = is_point_inside_polygon(point[0], point[1], polygon)
print("Is the point inside the polygon?", is_inside)
print(polygon)

# relationship between a point and a polygon
def is_point_inside_polygon(x, y, polygon):
    n = len(polygon)
    wn = 0  # Initialize winding number

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]  # Wrap around for the last edge

        if y1 <= y:
            if y2 > y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0:
                wn += 1
        elif y2 <= y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0:
            wn -= 1

    return wn != 0  # Point is inside if winding number is not zero

# generated_grids[0]
for grid in generated_grids:
  # get the vertices of the square
  square_vertices = grid[:4]
  center_x, center_y = grid[4]

  #  we have to check if the center x and y is inside a polygon
  for c_coords in non_overlap_aggregates:
    is_inside = is_point_inside_polygon(center_x, center_y, c_coords[2])
    if is_inside:
      grid[5] = True
    else:
      grid[7] = True

print(generated_grids)

"""Aggregates only plot"""

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

plt.figure()

for c_coords in non_overlap_aggregates:

    x_coords, y_coords = c_coords[0], c_coords[1]

    plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Aggregates only plot')
plt.grid(True)

if x_limit is not None:
    plt.xlim(x_limit)
if y_limit is not None:
    plt.ylim(y_limit)

plt.show()

"""Aggregates with grids plot"""

import matplotlib.pyplot as plt
import numpy as np


def plot_colored_cells():
    # Define the size of each cell
    cell_size = 5
    num_cells = 500 // cell_size

    # Create a new figure
    plt.figure()

    # Create axis
    ax = plt.gca()

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)

    # Create gridlines for the cells
    for x in range(0, 501, cell_size):
        ax.axvline(x, color='black', linestyle='-', linewidth=1)

    for y in range(0, 501, cell_size):
        ax.axhline(y, color='black', linestyle='-', linewidth=1)

    # Fill each cell with a color
    colors = np.random.rand(num_cells, num_cells, 3)  # Random colors for each cell
    for i in range(num_cells):
        for j in range(num_cells):
            # cell_color = colors[i, j, :]
            ax.fill_between([i * cell_size, (i + 1) * cell_size], [j * cell_size, j * cell_size], [(j + 1) * cell_size, (j + 1) * cell_size], color='white')


    for c_coords in non_overlap_aggregates:
      x_coords, y_coords = c_coords[0], c_coords[1]
      plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Aggregates with grids plot')

    # Display the plot
    plt.grid(True)
    plt.show()

# Call the function to generate the plot
plot_colored_cells()

"""Aggregate with coloured grids"""

import matplotlib.pyplot as plt
import numpy as np


def plot_colored_cells():
    # Define the size of each cell
    cell_size = 5
    num_cells = 500 // cell_size

    # Create a new figure
    plt.figure()

    # Create axis
    ax = plt.gca()

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)

    # Create gridlines for the cells
    for x in range(0, 501, cell_size):
        ax.axvline(x, color='black', linestyle='-', linewidth=1)

    for y in range(0, 501, cell_size):
        ax.axhline(y, color='black', linestyle='-', linewidth=1)

    # Fill each cell with a color
    colors = np.random.rand(num_cells, num_cells, 3)  # Random colors for each cell
    for i in range(num_cells):
        for j in range(num_cells):
            # cell_color = colors[i, j, :]
            if i == 0 and j  <= 2:
              ax.fill_between([i * cell_size, (i + 1) * cell_size], [j * cell_size, j * cell_size], [(j + 1) * cell_size, (j + 1) * cell_size], color='white')
            else:
              ax.fill_between([i * cell_size, (i + 1) * cell_size], [j * cell_size, j * cell_size], [(j + 1) * cell_size, (j + 1) * cell_size], color='green')


    for c_coords in non_overlap_aggregates:
      x_coords, y_coords = c_coords[0], c_coords[1]
      plt.fill(x_coords, y_coords, facecolor='red', edgecolor='white')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Aggregate with coloured grids')

    # Display the plot
    plt.grid(True)
    plt.show()

# Call the function to generate the plot
plot_colored_cells()

"""Material Identification of aggregates only using the background grid method without the actual visualization of the aggregates."""

import matplotlib.pyplot as plt
import numpy as np


def plot_colored_cells():
    # Define the size of each cell
    cell_size = 5
    num_cells = 500 // cell_size

    # Create a new figure
    plt.figure()

    # Create axis
    ax = plt.gca()

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)

    # Create gridlines for the cells
    for x in range(0, 501, cell_size):
        ax.axhline(x, color='black', linestyle='-', linewidth=1)

    for y in range(0, 501, cell_size):
        ax.axvline(y, color='black', linestyle='-', linewidth=1)

    # Fill each cell with a color
    colors = np.random.rand(num_cells, num_cells, 3)  # Random colors for each cell
    n = 0
    m = n
    for i in range(num_cells):
        for j in range(num_cells):
            # cell_color = colors[i, j, :]
          if generated_grids[m][5]:
            coloriser =  'red'
          else:
            coloriser =  'white'

          vertices = [
            (((i+1)-1)*cell_size, ((j+1)-1)*cell_size),
            ((i+1)*cell_size, ((j+1)-1)*cell_size),
            ((i+1)*cell_size, (j+1)*cell_size),
            (((i+1)-1)*cell_size, (j+1)*cell_size)
          ]

          # ax.fill_between([(i-1)*ecell, (j-1)*cell_size], [i*cell_size, (j-1)*cell_size], [i*cell_size, j*cell_size], color=coloriser)
          ax.fill_between([v[0] for v in vertices], [v[1] for v in vertices], color=coloriser)
          m += 100
        n += 1
        m = n

    # for c_coords in non_overlap_aggregates:
    #   x_coords, y_coords = c_coords[0], c_coords[1]
    #   plt.fill(x_coords, y_coords, facecolor='white', edgecolor='white')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Material Identification of aggregates only')

    # Display the plot
    plt.grid(True)
    plt.show()

# Call the function to generate the plot
plot_colored_cells()

"""Material Identification of aggregates, matrix and interfacial transition zone only using the background grid method without the actual visualization of the aggregates."""

import matplotlib.pyplot as plt
import numpy as np


def plot_colored_cells():
    # Define the size of each cell
    cell_size = 5
    num_cells = 500 // cell_size

    # Create a new figure
    plt.figure()

    # Create axis
    ax = plt.gca()

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)

    # Create gridlines for the cells
    for x in range(0, 501, cell_size):
        ax.axhline(x, color='black', linestyle='-', linewidth=1)

    for y in range(0, 501, cell_size):
        ax.axvline(y, color='black', linestyle='-', linewidth=1)

    # Fill each cell with a color
    colors = np.random.rand(num_cells, num_cells, 3)  # Random colors for each cell
    n = 0
    m = n
    for i in range(num_cells):
        for j in range(num_cells):
            # cell_color = colors[i, j, :]
          if generated_grids[m][5]:
            coloriser =  'red'
          elif generated_grids[m][7]:
            coloriser =  'green'
          else:
            coloriser =  'white'

          vertices = [
            (((i+1)-1)*cell_size, ((j+1)-1)*cell_size),
            ((i+1)*cell_size, ((j+1)-1)*cell_size),
            ((i+1)*cell_size, (j+1)*cell_size),
            (((i+1)-1)*cell_size, (j+1)*cell_size)
          ]

          # ax.fill_between([(i-1)*ecell, (j-1)*cell_size], [i*cell_size, (j-1)*cell_size], [i*cell_size, j*cell_size], color=coloriser)
          ax.fill_between([v[0] for v in vertices], [v[1] for v in vertices], color=coloriser)
          m += 100
        n += 1
        m = n

    for c_coords in non_overlap_aggregates:
      x_coords, y_coords = c_coords[0], c_coords[1]
      plt.fill(x_coords, y_coords, facecolor='red', edgecolor='white')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Material Identification of Aggregates, Matrix and ITZ')

    # Display the plot
    plt.grid(True)
    plt.show()

# Call the function to generate the plot
plot_colored_cells()

