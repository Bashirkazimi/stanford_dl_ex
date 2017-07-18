function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  n = size(X,1);
  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
  %%% YOUR CODE HERE %%%  
  for i=1:m
    prediction = 0;  
    for j=1:n
      prediction = prediction+theta(j)*X(j,i);
    end
    prediction = 1/(1+exp(-prediction));
    if y(i) == 0
       f = f+log(1-prediction); 
    else
       f = f+log(prediction);
    end
    temp = prediction - y(i);
    for j=1:n
        g(j) = g(j) + temp*X(j,i);
    end
  end
  f = (-1.0/(m))*f;

 