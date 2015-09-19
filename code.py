import numpy as np
import math as m
from scipy import stats
from scipy.special import (polygamma, psi,  gammaln)
import sys
import matplotlib.pyplot as plt
from pylab import *

def two_c(mu=5.0, sigma=1.0, obs=100, n=100):
	# delta method
	samples_x = np.random.normal(mu, sigma, obs)
	sample_x_mean = sum(samples_x) / len(samples_x)
	sample_x_sd = np.std(samples_x) / m.sqrt(obs)
	sample_theta_sd = np.exp(sample_x_mean) * sample_x_sd
	c_interval_delta = stats.norm.interval(0.95, loc=np.exp(sample_x_mean),
		scale=sample_theta_sd)
	print 'delta mean is ' + str(np.exp(sample_x_mean))
	print 'delta sd is ' + str(sample_theta_sd)
	print 'interval delta method is ' + str(c_interval_delta)

	# bootstrap method
	bootstrap_vals = []
	for boostrap_trial in range(n):
		samples_x = np.random.normal(mu, sigma, obs)
		bootstrap_sample_mean = sum(samples_x) / len(samples_x)
		bootstrap_vals.append(np.exp(bootstrap_sample_mean))
	bootstrap_sd = np.std(bootstrap_vals)
	bootstrap_mean = np.mean(bootstrap_vals)
	c_interval_bootstrap = stats.norm.interval(0.95, loc=bootstrap_mean,
		scale=bootstrap_sd)
	print 'bootstrap_mean is ' + str(bootstrap_mean)
	print 'bootstrap sd is ' + str(bootstrap_sd)
	print 'bootstrap interval is ' + str(c_interval_bootstrap)


def inverse_digamma(y, iters=6):
	# initialization taken from appendix c of
	# http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf
	x_old = np.piecewise(y, [y >= -2.22, y < -2.22],
		[(lambda x: np.exp(x) + 0.5), (lambda x: -1 / (x - psi(1)))])
	for i in range(iters):
		x_new = x_old - (psi(x_old) - y) / polygamma(1, x_old)
		x_old = x_new
	return x_old


def three_b(samples=np.random.dirichlet((10, 5, 3), 3000), iters=1000, for_plot=False):
	num_samples, k = samples.shape
	log_x_mean = np.log(samples).mean(axis=0)
	# unclear what to initialize to, but this seems to work well
	a_old = [0.1 for i in range(k)]
	a_by_iteration = []
	for i in xrange(iters):
		if for_plot:
			liklihood = 1.0
			i = 0
			for sample in samples:
				# The data you gave doesn't have enough precision for scipy to accept
				# it as a valid simplex. Sometimes it throws an exception that the data
				# sums to 0.999999 instead of 1.0. I exclude these exceptional cases
				try: 
					liklihood *= stats.dirichlet.logpdf(sample, a_old)
				except:
					continue
			a_by_iteration.append(liklihood)
		a_new = inverse_digamma(log_x_mean + psi(sum(a_old)))
		a_old = a_new
	conjecture = psi(a_old) - psi(sum(a_old))
	if not for_plot:
		print 'Conjecture is ' + str(conjecture) + ' log sample mean is ' + str(log_x_mean)
		return a_old
	return a_by_iteration

def three_c_i(data):
	x = map(lambda x: x[0], data)
	y = map(lambda x: x[1], data)
	fig = plt.figure()
	plt.scatter(x, y)
	ax = fig.add_subplot(111)
	ax.set_xlabel('first dimension')
	ax.set_ylabel('second dimension')
	ax.set_title('first and second dimensions (third dimension implied)')
	plt.show()

def three_c_ii(data):
	a_by_iteration = three_b(samples=np.array(data), iters=150, for_plot=True)
	fig = plt.figure()
	plt.scatter(range(len(a_by_iteration)), a_by_iteration)
	ax = fig.add_subplot(111)
	ax.set_xlabel('iteration')
	ax.set_ylabel('log likelihood')
	ax.set_title('log likelihood by iteration')
	plt.show()

def three_c_iii(data):
	x = map(lambda x: x[0], data)
	y = map(lambda x: x[1], data)
	fig = plt.figure()
	plt.scatter(x, y, color='green')
	ax = fig.add_subplot(111)
	ax.set_xlabel('first dimension')
	ax.set_ylabel('second dimension')
	ax.set_title('first and second dimensions (third dimension implied)')
	mle_a = three_b(samples=np.array(data), iters=500, for_plot=False)
	samples = np.random.dirichlet(mle_a, 1000000)
	samples_x = map(lambda x: x[0], samples)
	samples_y = map(lambda x: x[1], samples)
	hist2d(samples_x, samples_y, bins=100)
	show()


# set c_i, c_ii, c_iii=True to see the plots associated with these components
def three_c(fname='dir1.txt', c_i=False, c_ii=False, c_iii=True):
	f = open(fname)
	# read into array of arrays, convert strings to floats, trim new lines and trailing spaces
	data = map(lambda x: [float(val) for val in x], [line.split(' ')[:-1] for line in f.read().splitlines()])
	if c_i:
		three_c_i(data)
	if c_ii:
		three_c_ii(data)
	if c_iii:
		three_c_iii(data)
	


