function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X=[ones(m,1) X];
z2=X*Theta1';
a2=sigmoid(z2);
a2=[ones(m,1),a2];
z3=a2*Theta2';
a3=sigmoid(z3);

Y=[];
E=eye(num_labels);
for i=1:num_labels
  AD=find(y==i);
  Y(AD,:)=repmat(E(i,:),size(AD),1);
end

temp1=Theta1(:,2:end);
temp1=temp1.^2;
temp2=Theta2(:,2:end);
temp2=temp2.^2;

cost=-1/m*(Y.*log(a3)+(1-Y).*log(1-a3));
J=sum(cost(:))+lambda/(2*m)*(sum(temp1(:))+sum(temp2(:)));

err3=a3-E(y,:);
err2=err3*Theta2(:,2:end).*sigmoidGradient(z2);

delter2=err3'*a2;
delter1=err2'*X;

Theta1_grad=1/m*(delter1+lambda*[zeros(size(Theta1),1) Theta1(:,2:end)]);
Theta2_grad=1/m*(delter2+lambda*[zeros(size(Theta2),1) Theta2(:,2:end)]);

grad=[Theta1_grad(:);Theta2_grad(:)];

end