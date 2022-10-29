import numpy as np

ls_samples  = [1e+3, 1e+4, 1e+5, 1e+6]
ls_features = [5e+3, 1e+4, 5e+4, 1e+5, 5e+5]
​
# Top-down design
​
ls_benchmark = []
# we will have 20 scenarios to benchmark the performance in acc and time
for i in ls_samples:
    for j in ls_features:
        # we only alter two variables: # of samples and # of features
        benchmark = run_simulation(samples=i, features=j)
        # concatenate the results (benchmark) into the final list 'ls_benchmark'
        ls_benchmark += [benchmark]
​
​
​
def run_simulation(samples, features):
    # n = samples; m = features
    n = samples;
    m = features;
    #             y     =        Xb      + e
    # dimensions: n x 1 = {n x m}{m x 1} + n x 1
​
    ### simulate dataset

    # initialize hypothesis, beta, is a vector: {b0, b1, ..., bm}
    beta = simulate_beta(m=features)
    # simulate X
    X = simulate_beta(int(n),int(m))
    # simulate residual and y
    #e = simulate_beta(int(n),1) #?
    #Y = simulate_beta(int(n),1) #? Y = Xb + e
    # split data set using k-fold validation

    data_set = []
    k=5
    for i in range(k):
        tmp = []
        j = i
        while j < len(X):
            tmp.append(X[j])
            j = j + k
            data_set.append(tmp)

    for i in range(k):
        X_test = data_set[i]
        X_train = []
        for j in range(k):
            if i! = j:
                X_tain.append(data_set[j])
        print()
        print("processing fold #", i + 1)

        X_train =

​
    ### SGD

    import numpy as numpy
    x = np.arange(0., 10., 0.2)
    m = len(x)
    x0 = np.full(m, 1.0)
    print(x0)

    input_data = np.vstack([x. x0]).T

    # initialize b
    b = np.random.randn(2,1)

    for i in range(0,1000)

    error = 0
    while (error - error_tmp > 1e+3):
        error_tmp = error
        # calculate j'(b) for each b
        # update each b
    # keep the error and elapsed time as the benchmark
​
​
    ### Normal equation
    # estimate b using training dataset
    # validate b using testing dataset
    # keep the error and elapsed time as the benchmark
​
    return 0
​
​
def simulate_beta(m):
    """
    return a vector of hypothesis with length of m
    """
    return 0