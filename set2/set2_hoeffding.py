# Claire Goeckner-Wald
# CS156a set 2
# Hoeffding Inequality

from __future__ import division
import random

def flip_many_coins(num_coins, num_flips_per_coin):

	# 0 represents tails
	# 1 represents heads
	# Thus, a '3' in coin_arr index indicates that that coin landed on heads
	# 3 times out of the total number of times it was flipped (10)

	def flip_coin(num_flips_per_coin):
		sum_flips = 0
		for j in xrange(num_flips_per_coin):
			sum_flips += random.choice([0,1]) 
		return sum_flips

	def get_coin_data(num_coins, num_flips_per_coin):
		coin_arr = []
		minimum_heads = 11
		minimum_heads_index = -1
		for i in xrange(num_coins):
			# flip the coin num_flips_per_coin time to get the num_heads
			num_heads = flip_coin(num_flips_per_coin)
			# Mark which coin has the minimum number of heads
			# Note that this is not <=, because we want the earliest such 
			# coin with minimum heads
			if (num_heads < minimum_heads):
				minimum_heads = num_heads
				minimum_heads_index = i
			coin_arr.append(num_heads)
			num_heads = 0
		return (coin_arr, minimum_heads_index)

	coin_arr, minimum_heads_index = get_coin_data(num_coins, num_flips_per_coin)
	
	# The first coin flipped
	v_1 = coin_arr[0]/num_flips_per_coin
	# A random coin flipped
	v_rand = random.choice(coin_arr)/num_flips_per_coin
	# The coin with the minimum number of heads flipped
	v_min = coin_arr[minimum_heads_index]/num_flips_per_coin

	return (v_1, v_rand, v_min)

def experiment(num_repititions):
	distribution_array = [0, 0, 0]
	num_coins = 1000
	num_flips_per_coin = 10
	for k in xrange(num_repititions):
		temp_arr = flip_many_coins(num_coins, num_flips_per_coin)
		distribution_array[0] += temp_arr[0]
		distribution_array[1] += temp_arr[1]
		distribution_array[2] += temp_arr[2]

	distribution_array[0] = distribution_array[0]/num_repititions
	distribution_array[1] = distribution_array[1]/num_repititions
	distribution_array[2] = distribution_array[2]/num_repititions

	print distribution_array

# DO THE EXPERIMENT
num_repititions = 100000
experiment(num_repititions)

# Output for 100,000 repititions:
# [0.49965599999999977, 0.5002000000000099, 0.03761899999997675]


