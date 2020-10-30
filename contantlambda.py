import random
import math

_lambda = 5
_num_total_arrivals = 150
_num_arrivals = 0
_arrival_time = 0
_num_arrivals_in_unit_time = []
_time_tick = 1

print('RANDOM_N,INTER_ARRIVAL_TIME,EVENT_ARRIVAL_TIME')

for i in range(_num_total_arrivals):
	#Get the next probability value from Uniform(0,1)
	p = random.random()

	#Plug it into the inverse of the CDF of Exponential(_lamnbda)
	_inter_arrival_time = -math.log(1.0 - p)/_lambda

	#Add the inter-arrival time to the running sum
	_arrival_time = _arrival_time + _inter_arrival_time

	#Increment the number of arrival per unit time
	_num_arrivals = _num_arrivals + 1
	if _arrival_time > _time_tick:
		_num_arrivals_in_unit_time.append(_num_arrivals)
		_num_arrivals = 0
		_time_tick = _time_tick + 1

	#print it all out
	print(str(p)+','+str(_inter_arrival_time)+','+str(_arrival_time))

print('\nNumber of arrivals in successive unit length intervals ===>')
print(_num_arrivals_in_unit_time)

print('Mean arrival rate for sample:' + str(sum(_num_arrivals_in_unit_time)/len(_num_arrivals_in_unit_time)))