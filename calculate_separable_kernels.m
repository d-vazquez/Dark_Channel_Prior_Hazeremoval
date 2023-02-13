% calculate a kernel separate vectors,
% a Kernel s may be the outer product of 2 vectors, h and v
% meaning s = h * v, 
% convolution assosiative property allow us to obtain the same
% result from convoluting the s kernel (2D) by applying separatelly
% vectors h and v, os if I is the original image, and s is the kernel 
% the resulting convoluted image J is:
% J = I (*) s = ((I (*) h) (*) v)
% We save computaional time because we change a 2D multiplication for'
% 2 1D multiplication and a sum, for example,
% To convolute 1 pixel from a 3x3 kernel, we need to do 9 multiplications 
% and 1 addition, but if using separable vectors, then we do 
% 6 multiplication and 2 sums.

% When calculating separable kernels we notice that if the kernel is
% squared and seprable, the values on the horizontal and vertical vectors
% are always the same, so we want to store them in a LUT so we will only
% support square kernels

clear all;

% Kernel side
k_width  = 60;
k_height = k_width;


% Calculate kernel, box kernel formula isL
box = ones(k_width,k_height) / (k_width*k_height);

% Check if kernel is separable
if(rank(box) ~= 1)
    error("Kernel not separable, rank is not equal to 1")
end 

% get singular value decomposition
[U,S,V] = svd(box);

% Vectors v and h, usualy vector with same value
v = U(:,1)  * sqrt(S(1,1))
h = V(:,1)' * sqrt(S(1,1))


