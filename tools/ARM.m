function [R,S] = ARM(J,sigma_s, sigma_r,t)
I = mat2gray(J);
R = I;
for i = 1:t
   R = RF(R, sigma_s, sigma_r, 3, I);
end
S = I./max(R,1e-6);
