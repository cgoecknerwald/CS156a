import math

def change_u(u,v):
	return 2*(u*math.exp(v)-2*v*math.exp(-u))*(math.exp(v)+2*v*math.exp(-u))

def change_v(u,v):
	return 2*(u*math.exp(v)-2*v*math.exp(-u))*(u*math.exp(v)-2*math.exp(-u))

def error(u,v):
	return (u*math.exp(v)-2*v*math.exp(-u))**2

def experiment(u, v, eta, stop):
	iterations = 0
	err = error(u, v) 
	while (err > stop):
		du = change_u(u, v)
		dv = change_v(u, v)
		u = u - du*eta
		v = v - dv*eta
		err = error(u, v)
		iterations += 1
	return (iterations, u, v, err)

def experiment_coord(u, v, eta, stop):
	iterations = 0
	while (iterations < stop):
		# change u 
		du = change_u(u, v)
		u = u - du*eta
		# change v
		dv = change_v(u, v)
		v = v - dv*eta
		iterations += 1
	err = error(u, v)
	return (iterations, u, v, err)
		


print experiment(1,1,0.1,10**(-14))
print experiment_coord(1,1,0.1, 15)

# OUTPUT:
# (10, 0.04473629039778207, 0.023958714099141746, 1.2086833944220747e-15)
# (15, 6.29707589930517, -2.852306954077811, 0.13981379199615315)


