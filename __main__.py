'''
Basic Neural Network to indentify handwritten images from the mnist database
Author: Steven L
'''
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# defines the structure of the neural network
NODE_NUM = np.array([784, 20, 20, 10])
LAYER_NUM = len(NODE_NUM)

def load_pixels(filename):
    ''' load raw pixel data from given file '''
    im = Image.open(filename)
    pix = im.load()
    print(im.size)
    total_pixels = im.size[0] * im.size[1]

    index = 0
    pixels = np.zeros(total_pixels)
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            pixels[index] = pix[i,j]
            index += 1

    return pixels

def load_data(filename):
    ''' loads data from file and returns parsed edge and bias data '''
    with open(filename) as f:
        lines = f.readlines()
        edges = lines[0]
        biases = lines[1]
    edges = parse_edges(edges)
    biases = parse_biases(biases)
    return [edges, biases]

def save_data(edges, biases, filename):
    ''' converts edges and bias data into strings which are saved to file '''
    edges = flatten_data(edges)
    biases = flatten_data(biases)
    with open(filename, 'w') as f:
        f.write(','.join([str(edge) for edge in edges]) + '\n')
        f.write(','.join([str(bias) for bias in biases]) + '\n')

def flatten_data(data):
    return np.concatenate([d.flatten() for d in data])

def parse_edges(edge_str):
    ''' converts the string of edge data as saved '''
    edges_array = np.array([float(edge) for edge in edge_str.split(',')])
    edges = [0] * (LAYER_NUM - 1)
    for i in range(LAYER_NUM - 1):
        previous_size = 0 if i == 0 else NODE_NUM[i-1]*NODE_NUM[i]
        current_size = NODE_NUM[i]*NODE_NUM[i+1]
        edges[i] = edges_array[previous_size:previous_size+current_size].reshape(NODE_NUM[i+1], NODE_NUM[i])
    return edges

def parse_biases(biases_str):
    biases_array = np.array([float(bias) for bias in biases_str.split(',')])
    biases = [0] * (LAYER_NUM - 1)
    for i in range(LAYER_NUM - 1):
        previous_size = 0 if i == 0 else NODE_NUM[i]
        current_size = NODE_NUM[i+1]
        biases[i] = biases_array[previous_size:previous_size+current_size][:, np.newaxis]
    return biases

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def generate_random_data():
    ''' generate random edge and bias values in the shape given by NODE_NUM '''
    edges = [0] * (LAYER_NUM - 1)
    biases = [0] * (LAYER_NUM - 1)
    for i in range(LAYER_NUM - 1):
        edges[i] = np.random.rand(NODE_NUM[i+1], NODE_NUM[i]) * 2 - 1
        biases[i] = np.random.rand(NODE_NUM[i+1])[:, np.newaxis] * 2 - 1
    return [edges, biases]

def neural_compute(input, edges, biases):
    ''' returns the values of the output nodes for a given input '''
    input = input[:, np.newaxis]
    for i in range(LAYER_NUM - 1):
       input = sigmoid((edges[i] @ input) + biases[i])
    return input

def neural_state(input, edges, biases):
    ''' returns a list with the values of all nodes for given input node values '''
    input = input[:, np.newaxis]
    node_state = [input]
    for i in range(LAYER_NUM - 1):
        input = (edges[i] @ input) + biases[i]
        node_state.append(input)
        input = sigmoid(input)
    return node_state

def neural_rate(edges, biases):
    ''' gives the average cost function value of the test data '''
    sum_error = 0
    for test, y in zip(test_X, test_y):
        sum_error += neural_test(y, normalise(flatten_data(test)), edges, biases)
    return sum_error/len(test_X)

def neural_correct(edges, biases):
    ''' gives the amount of correct guess of the test data as a percentage '''
    correct = 0
    for test, y in zip(test_X, test_y):
        output = list(neural_compute(normalise(flatten_data(test)), edges, biases))
        if (output.index(max(output)) == y):
            correct += 1
    return (correct / len(test_X)) * 100

def neural_test(expected, input, edges, biases):
    ''' returns the value of the cost function for given input and an expected value '''
    expected_arr = expected_array(expected)
    output = neural_compute(input, edges, biases)
    error = np.sum(np.square(output - expected_arr))
    return error

def expected_array(index):
    ''' returns the numpy array that is expected as the output of the neural network using an index '''
    array = np.zeros(NODE_NUM[-1])[:, np.newaxis]
    array[index, 0] = 1
    return array

def get_empty_data():
    ''' returns empty biases and edges lists'''
    edges = [0] * (LAYER_NUM - 1)
    biases = [0] * (LAYER_NUM - 1)
    for i in range(LAYER_NUM - 1):
        edges[i] = np.empty((NODE_NUM[i], NODE_NUM[i+1], 1))
        biases[i] = np.empty((NODE_NUM[i+1], 1))
    return [edges, biases]

def cost_derivative(edges, node_state, biases, expected):
    ''' 
    returns matrices of the derivatives of the cost function with respect to the weights
    and biases in the form of lists of numpy arrays in the shape of edges and biases
    '''
    derivatives = [2 * (sigmoid(node_state[-1]) - expected)]
    
    for i in range(LAYER_NUM-2, 0, -1):
        derivatives.append(np.dot(edges[i].T, derivatives[-1]  * sigmoid_prime(node_state[i+1])))

    derivatives_m = []
    for i in range(len(derivatives)):
        layer_num = LAYER_NUM - 2 - i
        derivatives[i] *= sigmoid_prime(node_state[layer_num+1])
        derivatives_m.append(derivatives[i] @ node_state[layer_num].T)
    
    derivatives.reverse()
    derivatives_m.reverse()
    
    return derivatives_m, derivatives

def neural_train(edges, biases, train_X, train_y, start, batch):
    ''' averages results over batch iterations of cost_derivative for biases and weights '''
    train_X, train_y = train_X[start:start+batch], train_y[start:start+batch]
    derivatives_m = []
    derivatives_b = []
    i = 0
    for train, expected in zip(train_X, train_y):
        train = normalise(flatten_data(train))
        node_state = neural_state(train, edges, biases)
        der_e, der_b = cost_derivative(edges, node_state, biases, expected_array(expected))
        if i==0:
            derivatives_m = der_e
            derivatives_b = der_b
        else:
            for i in range(len(derivatives_m)):
                derivatives_m[i] += der_e[i]
                derivatives_b[i] += der_b[i]
        i += 1

    for der_m, edge_m in zip(derivatives_m, edges):
        der_m /= (batch * 100)
        edge_m -= der_m
    
    for der_b, bias in zip(derivatives_b, biases):
        der_b /= (batch * 100)
        bias -= der_b
    
    return edges, biases


def r_normalise(data):
    ''' normalise then reverse the values (0->1, 1->0, 0.5->0.5 etc) '''
    return 1 - (data - data.min())/(data.max())

def normalise(data):
    return (data - data.min())/(data.max())

def print_output(output):
    ''' prints output of neural_output to list top 3 results'''
    output = list(output.T[0])
    output_sorted = sorted(output)
    for i in range(1, 4):
        print(f"{i}: {output.index(output_sorted[-i])}")

def main():
    # edges, biases = generate_random_data()
    edges, biases = load_data('data.txt')

    # node_state = neural_state(np.array([0, 1, 0, 1]), edges, biases)
    # cost_derivative(edges, node_state, biases, expected_array(1))
    

    # pixels = load_pixels('datasets/my_test/what_is_it.png')
    # output = neural_compute(r_normalise(pixels), edges, biases)

    # example = 129

    # output = neural_compute(normalise(flatten_data(test_X[example])), edges, biases)
    # plt.imshow( test_X[example], cmap = 'Greys' , interpolation = 'none')

    # plt.show()

    # print(test_y[example])
    # print_output(output)

    

    print(f"Before: {neural_rate(edges, biases)}")
    print(f"Correct: {neural_correct(edges, biases)}")

    for j in range(25):
        print(j)
        for i in range(0, 60001, 1000):
            edges, biases = neural_train(edges, biases, train_X, train_y, i, 1000)
    
    print(f"After: {neural_rate(edges, biases)}")
    print(f"Correct: {neural_correct(edges, biases)}")

    save_data(edges, biases, 'data.txt')

main()
    




