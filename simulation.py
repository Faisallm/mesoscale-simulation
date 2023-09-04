# method one of generation and packing for
# take and place method + rotation procedure


# probability function that gives us the probability.
# of an (n)-sided aggregate occuring.
# omega = 2.2.
# nc = 5.8.

# imports
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# all units are in mm

# global variables
# this will store 50,000 generated aggregates
# each aggregate is a list of x_coords and y_coords
generated_aggregates = []


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

def check_area(dkl, dku):
  return (((dkl + dku)/4)**2) * math.pi


def is_polygon_valid(x_coords, y_coords):
    num_vertices = len(x_coords)
    
    if num_vertices < 3:
        return False
    
    def count_intersections(x, y):
        count = 0
        for i in range(num_vertices):
            x1, y1 = x_coords[i], y_coords[i]
            x2, y2 = x_coords[(i + 1) % num_vertices], y_coords[(i + 1) % num_vertices]
            
            if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
                if y1 != y2:
                    x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= x_intersect:
                        count += 1
        
        return count % 2
    
    # Check all vertices for odd number of intersections
    for i in range(num_vertices):
        if not count_intersections(x_coords[i], y_coords[i]):
            return False
    
    return True


def calculate_polygon_area(x_coords, y_coords):
    n = len(x_coords)
    
    if n < 3:
        return 0
    
    x_coords.append(x_coords[0])  # Closing the polygon
    y_coords.append(y_coords[0])
    
    area = 0
    for i in range(n):
        area += (x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i])
    
    return abs(area) / 2



def generate_aggregate():

  # calculate the cumulative probability of an aggregate passing through
  aggregate_size_range = [[0.15, 0.300], [0.300, 0.6], [0.6, 1.18],
    [1.18, 2.36], [2.36, 4.75], [4.75, 9.5]]

  # this will compute the cumulative probability that an aggregate..
  # passes a sieve opening with a size of d mm.
  probabilities = [fuller_curve(i[1]) for i in aggregate_size_range]
  print(f"Probabilities for aggregate_size_range: {probabilities}")

  # random.choices will randomly select an aggregate size range...
  # while considering that each size range has a weight value...
  # that determines the probability of its occurance.
  size = random.choices([i[1] for i in aggregate_size_range], weights = probabilities)

  range = [i for i in aggregate_size_range if i[1] == size[0]]
  print(range)
  dkl = range[0][0]
  dku = range[0][1]

  # calculate the first angle

  # number of sides of the aggregate as given in the paper.
  number_of_sides = [3, 4, 5, 6, 7, 8, 9]
  # array of the probability of an n-sided aggregate occuring.
  probabilities = [number_of_polygon_sides(i) for i in number_of_sides]

  # randomly picking the n-sides of an aggregates based on the probability function
  n = random.choices(number_of_sides, probabilities)[0]
  print(f"sides: {n}")

  # I now have my range of size and number of sides

  # calculate first angle and radius
  fst_angle, first_radius = first_angle(n, dkl, dku)

  # second angle and radius calculation
  angle_and_radius = []
  print(f'n: {n}, dkl: {dkl}, dku: {dku}')
  ith = 2
  while ith <= n:
    angle, radius = other_angle(n, dkl, dku)
    angle_and_radius.append([angle, radius])
    ith += 1

  angle_in_rads = []

  angle_and_radius.insert(0, (fst_angle, first_radius))

  # faisal
  for index, i in enumerate(angle_and_radius):
    if index == 0:
      angle_in_rads.append(i[0])
    else:
      sum_of_angle = sum([i[0] for i in angle_and_radius[:index]])
      angle_in_rads.append(i[0] + sum_of_angle)

  print(f"Angle in degrees: {angle_in_rads}")
  print(f"Angle and radius: {angle_and_radius}")

  x_coords = []
  y_coords = []

  for index, i in enumerate(angle_and_radius):
    x_coord = i[1] * np.cos(angle_in_rads[index])
    y_coord = i[1] * np.sin(angle_in_rads[index])

    x_coords.append(x_coord)
    y_coords.append(y_coord)

  print(f"length of x_coords: {len(x_coords)}")
  print(f"length of y_coords: {len(y_coords)}")
  print(f"length of angle radians: {len(angle_in_rads)}")

  area = calculate_polygon_area(x_coords, y_coords)
  print(f"The area of the polygon is {area}")

  area_check = check_area(dkl, dku)
  print(f"Required area of the polygon is {area_check}")

  # add generated aggregate to the list of aggregates
  generated_aggregates.append([x_coords, y_coords])
  print("Aggregate has been added to the database of aggregates")

  # if 0.5 * area_check <= area <= 1.5 * area_check:
  #   plt.figure()
  #   plt.plot(x_coords, y_coords, marker='o')
  #   plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 'k-')
  #   plt.xlabel('X-axis')
  #   plt.ylabel('Y-axis')
  #   plt.title('Polygon from Set of Angles')
  #   plt.grid(True)
  #   plt.axis('equal')
  #   plt.show()


# # calling the 
# if __name__ == "__main__":
for _ in range(1000):
  # generate 10 aggregates
  generate_aggregate()







# # number of sides of the aggregate as given in the paper.
# number_of_sides = [3, 4, 5, 6, 7, 8, 9]
# # array of the probability of an n-sided aggregate occuring.
# probabilities = [number_of_polygon_sides(i) for i in number_of_sides]

# # for 50,000 generated aggregates.
# # the number of n-sided aggregates out of the sample size
# print([probability*50000 for probability in probabilities])
# print(sum([probability*50000 for probability in probabilities]))





# packing of aggregates

import matplotlib.pyplot as plt
import random


# global variables
trial_locations = []

def generate_trial_location(square_size, num_of_trials=10):
  """This will generate the trial aggregate locations"""

  trial_locations = []
  for _ in range(num_of_trials):
    x = int(random.uniform(0, square_size))
    y = int(random.uniform(0, square_size))

    trial_locations.append((x, y))

  return trial_locations


def sort_aggregates(agregate_list):
  """This will arrange the aggregate is a descending
    order of size"""

  return sorted(agregate_list, key=lambda x: max(x), reverse=True)


def set_origin(aggregate, location):
  """This will place the aggregate at the generated
  aggregate location.
  aggregate: is a list containing [x_coords, y_coords]
  location: (x, y)"""

  # old version
  # aggregate[0].insert(0, location[0])   # x_coord
  # aggregate[1].insert(0, location[1])   # x_coord
  
  # new version
  # substracting x_coords
  # print(f"location: {location}")
  for index, x in enumerate(aggregate[0]):
    aggregate[0][index] -= location[0]

    # print(f"old: {x}, new: {aggregate[0][index]}")
  # substracting y_coords

  for index, y in enumerate(aggregate[1]):
    aggregate[1][index] -= location[1]
    # print(f"old: {x}, new: {aggregate[1][index]}")

  return aggregate

def place_aggregates():
  global trial_locations

  # this will return a list of locations
  trial_locations = generate_trial_location(1000, 1000);

  # sort aggregate by size (descending order)
  sorted_aggregates = sort_aggregates(generated_aggregates)

  # pop an aggregate out of the queue and place at...
  # a randomly selected trial position
  for aggregate, location in zip(sorted_aggregates, trial_locations):
    set_origin(aggregate, location)


  for i in range(len(sorted_aggregates)):
    print(f"Aggregates: {sorted_aggregates[i]}, location: {trial_locations[i]}")

  square_size = -1000

  # Create a figure and axis
  fig, ax = plt.subplots()

  for aggregate in sorted_aggregates:
    print(aggregate[0])
    ax.plot(aggregate[0], aggregate[1])

  # Set axis limits to the square area
  ax.set_xlim(0, square_size) 
  ax.set_ylim(0, square_size)

  # Set aspect ratio to ensure the square shape
  ax.set_aspect('equal')

  # Show the plot 
  plt.show()





place_aggregates()



# generate_trial_location(50, 10)

# sorting list of aggregates by maximum value


# for i in range(10):
#   print(calculate_polygon_area(sorted_aggregates[i][0], sorted_aggregates[i][1]))