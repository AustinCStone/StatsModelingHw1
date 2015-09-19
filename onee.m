function [ ] = onee()
% number 1 part e
p = 1000 ;
n = 20;
sets = 1000;
trials = sets * n;
mu = zeros(1,p);
sigma = eye(p);
samples = mvnrnd(mu, sigma, trials);
theta_hats = zeros(1,sets - 1);
theta_tildes = zeros(1,sets - 1);
theta_index = 1;
for i = 1:n:(trials - n) 
    theta_hats(theta_index) = norm(mean(samples(i:(i + n), :))) - p / n;
    theta_tildes(theta_index) = norm(mean(samples(i:(i + n), :))) + p / n;
    theta_index = theta_index + 1;
end
subplot(2,1,1);
hist(theta_hats);
title('Theta hats');
subplot(2,1,2);
hist(theta_tildes);
title('Theta tildes');

end

