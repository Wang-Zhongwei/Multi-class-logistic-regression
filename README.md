# Multi-class-logistic-regression

I built (n - 1) classifiers to classify i-th ‘梯队’ from other '梯队'. For Classifier activation function I chose f(x; w) = 1/(1 + exp(-w'.x)) where w'.x 
is the net_input, w is weights vector and x is the input vector. 

Results are stored in the .log file. Weights are normalized during iterations but were converted back to orginal scale afterwards (still shifted). Cost vs
iteration graphs are stored in the images folder. 
