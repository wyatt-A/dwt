signal = 1:20;

[lo_d,hi_d,lo_r,hi_r] = wfilters('db2')


sprintf("%1.60f,\n",double((hi_r)))

%%
x = 1:10;
n = 3;
[Lo_D,Hi_D,Lo_R,Hi_R]=wfilters('db2');

s = size(x);
x = x(:).'; % row vector
c = [];
l = zeros(1,n+2,'like',real(x([])));


l(end) = length(x);
for k = 1:n
    [x,d] = dwt(x,Lo_D,Hi_D); % decomposition
    c     = [d c];            %#ok<AGROW> % store detail
    l(n+2-k) = length(d);     % store length
end

% Last approximation.
c = [x c];
l(1) = length(x);

if s(1)>1
    c = c.'; 
    l = l';
end

l
c

%%


rmax = length(l);
% The bookkeeping vector has the number of resolution levels + 2 elements
% The number of coefficients in the coarsest wavelet details is repeated
% because of the approximation coefficients.
% The original signal length is included.
nmax = rmax-2;
n = nmax;
n=0
% Initialization. Find the approximation coefficients at the coarsest
% resolution.
a = c(1:l(1));

% Iterated reconstruction.
% The maximum reconstruction index is rmax+1 because the last element is
% the length of the input signal.
imax = rmax+1;
for p = nmax:-1:n+1
    d = detcoef(c,l,p);                % extract detail
    % Here we reverse the number of coefficients per resolution level
    l(imax-p)
    a = idwt(a,d,Lo_R,Hi_R,l(imax-p));
end

a







