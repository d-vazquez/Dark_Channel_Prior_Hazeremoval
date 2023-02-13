clc
clear all

A = [1     3     5     7     5;
     9     6     7     5     5;
     8     5     2     9     3;
     2     4     9     8     2;
     0     3     3     8     1;
     1     0     6     4     3;
     ]

box = ones(3,3) / (3*3);

A_padded = padarray(A,[1,1],'symmetric');

C = conv2(A_padded,box, 'valid')

% get separate kernel
[U,S,V] = svd(box);

v = U(:,1)  * sqrt(S(1,1));
h = V(:,1)' * sqrt(S(1,1));

C2 = conv2(A_padded,h,'valid');
C3 = conv2(C2,v,'valid')
