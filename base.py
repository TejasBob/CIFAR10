import numpy as np 
from scipy import signal

class Unit:
	def __init__(self, value, grad):
		self.value = value
		self.grad = grad

class Convolution:
	def __init__(self):
		self.u0 = 0
		self.u1 = 0

	def forward(self, u0, u1):
		self.u0 = u0
		self.u1 = u1

		rows, columns, depth = u0.shape[:3]
		for ch in range(depth):
			for i in range(rows):
				for j in range(columns):
					
		self.utop = Unit(np.sum(signal.convolve2d(self.u0, self.u1, boundary='symm', mode='same'), axis = 2), 0)
		return self.utop

	def backward(self):




class multiplyGate:
	def __init__(self):
		self.u0 = 0
		self.u1 = 0

	def forward(self, u0, u1):
		self.u0 = u0
		self.u1 = u1
		self.utop = Unit(self.u0.value * self.u1.value, 0)
		return self.utop

	def backward(self):
		self.u0.grad += self.u1.value * self.utop.grad # (local grad * grad_from_top)
		self.u1.grad += self.u0.value * self.utop.grad # (local grad * grad_from_top)


class addGate:
	def __init__(self):
		self.u0 = 0
		self.u1 = 0

	def forward(self, u0, u1):
		self.u0 = u0
		self.u1 = u1
		self.utop = Unit(u0.value + u1.value, 0)
		return self.utop

	def backward(self):
		self.u0.grad += self.utop.grad
		self.u1.grad += self.utop.grad

class sigmoidGate:
	def __init__(self):
		self.u0 =0
		

	def forward(self, u0):
		self.u0 = u0
		self.utop = Unit(1 / (1 + np.exp(-u0.value)), 0)
		return self.utop

	def backward(self):
		self.u0.grad += (self.utop.value * (1 - self.utop.value)) * self.utop.grad

def forward(mul0, mul1, add0, add1, sig0, a, x, b, y, c):
	ax = mul0.forward(a, x)
	by = mul1.forward(b,y)
	axpby = add0.forward(ax, by)
	axpbypc = add1.forward(axpby, c)
	s = axpbypc
	#s = sig0.forward(axpbypc)
	return s

def backward(s, sig0, add1, add0, mul1, mul0):
	s.grad = 1 #if (4 - s.value) >=0 else -1 # (expected - current_value)
	#print("grad : ", s.grad, end = "\t")
	#sig0.backward()
	add1.backward()
	add0.backward()
	mul1.backward()
	mul0.backward()
	return
def update(a,b,c,x,y):
	learning_rate = 1e-3

	a.value += learning_rate * a.grad
	b.value += learning_rate * a.grad
	c.value += learning_rate * c.grad
	# x.value += learning_rate * x.grad
	# y.value += learning_rate * y.grad
	return

def main():
	a = Unit(10, 0)
	b = Unit(2,0)
	c = Unit(3, 0)
	x = Unit(-1, 0)
	y = Unit(5,0)

	mul0 = multiplyGate()
	mul1 = multiplyGate()
	add0 = addGate()
	add1 = addGate()
	sig0 = sigmoidGate()

	# forward
	flag = 1
	counter = 0
	loss_array = []
	while(counter<1000):
		s = forward(mul0, mul1, add0, add1, sig0, a, x, b, y, c)
		s.value = round(s.value, 3)
		print("s : ", s.value, end="\t")
		loss_array.append(s.value)
		if s.value >=4:# and s.value >= 0.09:
			print("optima reached")
			break
		backward(s, sig0, add1, add0, mul1, mul0)
		update(a,b,c,x,y)
		counter+=1
	# plt.plot(range(len(loss_array)), loss_array)
	# plt.show()

	#print("Training completed")
		#print(a.grad, b.grad, c.grad, x.grad, y.grad)
		#print(forward(mul0, mul1, add0, add1, sig0, a, x, b, y, c).value)
	
	


if __name__ == '__main__':
	main()



