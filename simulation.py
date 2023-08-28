# method one of generation and packing for
# take and place method + rotation procedure


# probability function that gives us the probability.
# of an (n)-sided aggregate occuring.
# omega = 2.2.
# nc = 5.8.

# imports
import math
import random

# all units are in mm


def number_of_polygon_sides(n):
  """probability function for the number of polygon sides n."""

  # both the data below is obtained by fitting on the statistics
  omega = 2.2  # as given in the paper
  nc = 5.8

  return (1/(omega*math.sqrt(math.pi/2)))*math.exp((-2*(n-nc)**2)/omega**2)


def first_angle(n):
  """After defining an origin inside the aggregate. set the polar
  angle theta of the first corner"""
  nl = random.random()
  return (nl/n)*2*math.pi


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
  return ((d/9.5)**m)

# This function gets just one pair of coordinates based on the angle theta
def get_circle_coord(theta, x_center, y_center, radius):
    x = radius * math.cos(theta) + x_center
    y = radius * math.sin(theta) + y_center
    return (x,y)

def generate_aggregate():

  # calculate the cumulative probability of an aggregate passing through
  aggregate_size_range = [[0.15, 0.300], [0.300, 0.6], [0.6, 1.18], 
    [1.18, 2.36], [2.36, 4.75], [4.75, 9.5]]

  probabilities = [fuller_curve(i[1]) for i in aggregate_size_range]
  size = random.choices([i[1] for i in aggregate_size_range], probabilities)
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
  
  # I now have my range of size and number of sides

  # calculate first angle
  first_theta_angle = first_angle(n)

  # second angle calculation
  angle_and_radius = []
  print(f'n: {n}, dkl: {dkl}, dku: {dku}')
  ith = 2
  while ith <= n:
    angle, radius = other_angle(n, dkl, dku)
    angle_and_radius.append([angle, radius])
    ith += 1

  polygon = [get_circle_coord(i[0], 0, 0, i[1]) for i in angle_and_radius]
  
  print(angle_and_radius)


  

generate_aggregate()







# number of sides of the aggregate as given in the paper.
number_of_sides = [3, 4, 5, 6, 7, 8, 9]
# array of the probability of an n-sided aggregate occuring.
probabilities = [number_of_polygon_sides(i) for i in number_of_sides]

# for 50,000 generated aggregates.
# the number of n-sided aggregates out of the sample size
print([probability*50000 for probability in probabilities])
print(sum([probability*50000 for probability in probabilities]))
