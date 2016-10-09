# Claire Goeckner-Wald
# CS156a set 2
# Hoeffding Inequality

from __future__ import division
import random

def flip_1000_coins():

	# 0 represents tails
	# 1 represents heads
	# Thus, a '3' in coin_arr index indicates that that coin landed on heads
	# 3 times out of the total number of times it was flipped (10)

	def get_coin_arr():
		coin_arr = []
		num_coins = 1000
		num_flips = 10
		sum_flips = 0
		minimum_heads = 11
		minimum_heads_index = -1
		for i in xrange(num_coins):
			for j in xrange(num_flips):
				sum_flips += random.choice([0,1]) 
			# Mark which coin has the minimum number of heads
			# Note that this is not <=, because we want the earliest such 
			# coin with minimum heads
			if (sum_flips < minimum_heads):
				minimum_heads = sum_flips
				minimum_heads_index = i
			coin_arr.append(sum_flips)
			sum_flips = 0
		return (coin_arr, minimum_heads_index)

	temp_arr = get_coin_arr()
	coin_arr = temp_arr[0]
	minimum_heads_index = temp_arr[1]
	# The first coin flipped
	c_1 = coin_arr[0]
	v_1 = c_1/10

	# A random coin flipped
	c_rand = coin_arr[random.randrange(num_coins)]
	v_rand = c_rand/10

	# The coin with the minimum number of heads flipped
	c_min = coin_arr[minimum_heads_index]
	v_min = c_min/10

	return (v_1, v_rand, v_min)

def experiment(num_repititions):
	distribution_array = [0, 0, 0]
	for k in xrange(num_repititions):
		temp_arr = flip_1000_coins()
		distribution_array[0] += temp_arr[0]
		distribution_array[1] += temp_arr[1]
		distribution_array[2] += temp_arr[2]

	distribution_array[0] = distribution_array[0]/num_repititions
	distribution_array[1] = distribution_array[1]/num_repititions
	distribution_array[2] = distribution_array[2]/num_repititions

	print distribution_array

# DO THE EXPERIMENT
num_repititions = 100
experiment(num_repititions)

# Output for 100,000 repititions:
# [0.49965599999999977, 0.5002000000000099, 0.03761899999997675]
# [Finished in 640.0s]

