import math
import random

# For x in [-1,1] only
def f(x):
	return math.sin(math.pi*x)

# Get n random points from [-1, 1] returned in a list
def get_rand_points(n):
	return (random.uniform(-1,1) for i in xrange(n))

# Get the slope of the y = bx function
def get_function_slope():
	x1, x2 = get_rand_points(2)
	y1, y2 = f(x1), f(x2)
	slope = ((x1 * y1) + (x2 * y2)) / (x1**2 + x2**2)
	return slope

# Estimate the bias of the average function hypothesis over 1000 out-sample points
def estimate_bias(avg_slope):
	def g_avg(x):
		return avg_slope*x
	num_sample_pts = 1000
	sum_deviation = 0
	for x in get_rand_points(num_sample_pts):
		sum_deviation += (g_avg(x) - f(x))**2
	return sum_deviation/num_sample_pts

# Estimate the variance of the average function hypothesis from each individual hypothesis
def estimate_variance(N, avg_slope, slopes_arr):
	def g_avg(x):
		return avg_slope*x
	def g(slope,x):
		return slope*x
	num_sample_pts = 1000
	sum_deviation = 0
	# Find the deviation from g_avg over some samples points for each g
	for x in get_rand_points(num_sample_pts):
		for g_slope in slopes_arr:
			sum_deviation += (g_avg(x) - g(g_slope,x))**2
	# The variance is the average for each g and for some sample points
	return sum_deviation/N/num_sample_pts

# Returns (avg a, avg b, avg bias, avg variance)
def experiment(N):
	sum_slope = 0
	slopes_arr = []
	bias = 0
	variance = 0
	avg_b = 0
	for n in xrange(N):
		sl = get_function_slope()
		slopes_arr.append(sl)
		sum_slope += sl
	avg_slope = sum_slope/N
	return (avg_slope, avg_b, estimate_bias(avg_slope), estimate_variance(N, avg_slope, slopes_arr))

# RUN THE EXPERIMENT N TIMES
N = 100000
output = experiment(N)
print "average slope:", output[0], "\naverage y-intercept:", output[1], "\nbias:", output[2], "\nvariance:", output[3]

# EXAMPLE OUTPUT:
# average slope: 1.43271551919 
# average y-intercept: 0 
# bias: 0.262707944201 
# variance: 0.237766733032




