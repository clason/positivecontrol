function positivecontrol
%POSITIVECONTROL solves the elliptic control problem
% min 1/2 \|Ey-yd\|^2  s.t. -\Delta y = Bu , u\geq 0
% described in the paper
% "Optimal control of elliptic equations with positive measures"
% by Christian Clason and Anton Schiela, see
% http://math.uni-graz.at/mobis/publications/SFB-Report-2015-003.pdf.
%
% March 9, 2015              Christian Clason <christian.clason@uni-due.de>

N = 128;
zf = @(x,y) (abs(y-1/2)<1/4).*(abs(x-1/2)<1/4)+ ...
    0.5*(abs(y+1/2)<1/4).*(abs(x+1/2)<1/4); % z2 (blocks)
% zf = @(x,y) (1-x.^2).*(1-y.^2);
of = @(x,y) (abs(x)<=0.75).*(abs(y)<=0.75);  % box
% of = @(x,y) abs(x).^2+abs(y).^2 <= 0.75.^2;

%% setup
maxit = 20;  % max iterations in semismooth Newton method

% setup grid, assemble stiffness and mass matrix
[A,M,xx,yy] = assembleFEM(N);
tplot = @(n,f,s) tplot_(n,f,s,N,xx,yy);

% control domain
om = of(xx(:),yy(:));
B  = spdiags(of(xx(:),yy(:)),0,N*N,N*N); %=B'=B*B' since u defined on Omega

% target
yd = zf(xx(:),yy(:));

close all
figure(1);
tplot(1,yd,'target');
drawnow;

%% initialization
y = zeros(N*N,1); p = zeros(N*N,1);  As_old = zeros(N*N,1);

% continuation strategy
for alpha = 10.^(-(0:12))
    fprintf('\nSolving for alpha = %1.0e\n',alpha);
    fprintf('Iter\t update\t\t residual\n');
    
    % semismooth Newton iteration
    for iter = 0:maxit
        % compute active sets
        As = (-om.*p > 0);
        
        % compute gradient (note: As.*p = B*(As.*B'*p))
        b  = [A*y+(As.*p)/alpha; A*p-M*(y-yd)];
        
        % check termination
        update = nnz(As-As_old);  nres = norm(b);
        fprintf('%i\t %5d\t\t %1.3e\n', iter, update, nres);
        if update == 0 && nres < 1e-9
            break
        end
        
        % setup Newton step
        DN = spdiags(As/alpha,0,N*N,N*N); % = B*As*B since Am=0 outside om
        C  = [A DN; -M A];
        
        % solve for Newton step, update
        dx = -C\b;
        y  = y + dx(1:N*N);
        p  = p + dx(1+N*N:end);
        As_old = As;
    end
    
    % terminate continuation if Newton iteration converged in one step
    if iter <= 1
        break
    end
end

% compute starting point for control
u  = 1/alpha*max(0,-om.*p);

%% compute optimal measure space control
fprintf('\nSolving original problem\n');
fprintf('Iter\t update\t\t residual\n');

I = speye(N*N);
O = spalloc(N*N,N*N,0);

for iter = 0:maxit
    % compute active sets
    As = (u-om.*p > 0);
    
    % compute gradient
    b  = [A*y-B*u; A*p-M*(y-yd); u-As.*(u-B'*p)];
    
    % check termination
    update = nnz(As-As_old);  nres = norm(b);
    fprintf('%i\t %5d\t\t %1.3e\n', iter, update, nres);
    if update == 0 && nres < 1e-9
        break
    end
    
    % setup Newton step
    DN = spdiags(As,0,N*N,N*N);
    C  = [A O -B; -M A O; O DN I-DN];
    
    % solve for Newton step, update
    dx = -C\b;
    y  = y + dx(1:N*N);
    p  = p + dx(1+N*N:2*N*N);
    u  = u + dx(1+N*N*2:end);
    As_old = As;
end

%% plot results

% control
tplot(2,u,'control');
hold on;
contour(xx,yy,reshape(om,N,N),1);
hold off;
view(3);caxis([min(u),max(u)]);

% state
tplot(3,y,'state');

% sparsity pattern
up = u; up(u==0)=NaN;
figure(4);stem3(xx,yy,reshape(0.5+0*up,N,N),'MarkerSize',3); caxis([0,1]);
hold on;
contour(xx,yy,reshape(om,N,N),1);
hold off;
xlabel('x_1'); ylabel('x_2'); view(2);


function [K,M,xx,yy] = assembleFEM(n)
% assemble finite element matrices
nel = 2*(n-1)^2;         % number of nodes
h2  = (2/(n-1))^2;       % Jacobi determinant of transformation (2*area(T))

% nodes
[xx,yy] = meshgrid(linspace(-1,1,n));

% triangulation
tri = zeros(nel,3);
ind = 1;
for i = 1:n-1
    for j = 1:n-1
        node         = (i-1)*n+j+1;              % two triangles meeting at node
        tri(ind,:)   = [node node-1 node+n];     % triangle 1 (lower left)
        tri(ind+1,:) = [node+n-1 node+n node-1]; % triangle 2 (upper right)
        ind = ind+2;
    end
end

% Mass and stiffness matrices
Ke = 1/2 * [2 -1 -1 -1 1 0 -1 0 1]'; % elemental stiffness matrix
Me = h2/24 * [2 1 1 1 2 1 1 1 2]';   % elemental mass matrix

ent = 9*nel;
row = zeros(ent,1);
col = zeros(ent,1);
valk = zeros(ent,1);
valm = zeros(ent,1);

ind = 1;
for el=1:nel
    ll       = ind:(ind+8);             % local node indices
    gl       = tri(el,:);               % global node indices
    row(ll)  = gl([1;1;1],:); rg = gl';
    col(ll)  = rg(:,[1 1 1]);
    valk(ll) = Ke;
    valm(ll) = Me;
    ind      = ind+9;
end
M = sparse(row,col,valm);
K = sparse(row,col,valk);

% modify matrices for homogenenous Dirichlet conditions
bdnod = [find(xx == -1); find(yy == -1); find(xx == 1); find(yy == 1)];
M(bdnod,:) = 0;
K(bdnod,:) = 0;  K(:,bdnod) = 0;
for j = bdnod'
    K(j,j) = 1; %#ok<SPRIX>
end


function tplot_(n,f,s,N,xx,yy)
% plot finite element function
figure(n); surf(xx,yy,reshape(f,N,N)); axis tight
shading interp;lighting phong;camlight headlight;alpha(0.8)
title(s); xlabel('x_1'); ylabel('x_2');
drawnow;
