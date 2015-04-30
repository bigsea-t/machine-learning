import numpy as np
import math

a = np.array([1,3])
e = 5
B = np.array([1,-0.5,-0.5,3]).reshape(2,2) * 0.5
c = np.array([2,3])
d = 10
x_0 = np.array([4,4])

def fx(x):
	return np.exp(-np.dot(a,x) + e) + np.dot(np.dot(x,B),x) - np.dot(c,x) + d

def dfx(x):
	return a * np.exp(-np.dot(a,x) + e) + np.dot(x,B) - c

def ddfx(x):
	return np.outer(a,a) *	np.exp(-np.dot(a,x) + e) + B

def main():
	x_new = x_0
	x_old = None
	while(True):
		x_old = x_new
		invH  = np.linalg.inv(ddfx(x_old))
		x_new = x_old - np.dot(invH, dfx(x_old))
		diff  = math.fabs(np.linalg.norm(x_new-x_old))
		if(diff < 0.00001):
			break
	print('result:',x_new)

if __name__ == '__main__':
	main()