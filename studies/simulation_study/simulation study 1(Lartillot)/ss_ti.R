set.seed(42)
# To install powModSel R-package
devtools::install_github('pmat747/powModSel');
library(powModSel);
library(ggplot2)
library(patchwork)
##########################
### functions for SSTI ###
##########################
llike = function(theta, v){
  -sum(theta^2) / (2*v);
}
v = 0.01
p = 100
log.true.c <- function(v, p) {
  (p / 2) * (log(v) - log(1 + v)) # this formula is given right after (58) in Lartillot.
}
ltrue.c = log.true.c(v = v, p = p)
######################
# function to prepare for the matrix of log likelihoods and temperatures
f = function(k, n, v, d){
  
  # f returns a vector with the log-likelihoods and temperatures 
  # of n power posterior samples
  
  # k: number of temperatures
  # n: number of samples per temperature
  # v: variance in likelihood
  # d: dimension
  
  beta = matrix(qbeta(seq(0,1,length = k), 0.3, 1));
  
  pow_samples = apply(beta, 1, 
                      function(x){rnorm(d*n, mean=0, sd = sqrt(v/(v+x)))}); 
  pow_samples = matrix(c(pow_samples), byrow = TRUE, nrow = n * k, ncol = d);
  
  ls = apply(pow_samples, 1, function(y)llike(theta = y, v = v));
  
  ls = as.data.frame(cbind(ls, rep(beta, each = n)));
  colnames(ls) = c("logL", "invTemp")
  
  return(ls)
  
}



###################
### simulations ###
###################
ssti.results = matrix(nrow = 300, ncol = 2)
ptm <- proc.time()
for (i in 1:300){
  r = f(k=1000, n=1000, v=v, d=p)
  ssti.results[i, 1] = ss(r)
  ssti.results[i, 2] = ti(r)
  print(paste0("iteration", i))
}
(proc.time() - ptm) / 300

#################################################
### post processing of the simulation results ###
#################################################
(sim.mean = apply(ssti.results, 2, mean))
(sim.se = apply(ssti.results, 2, sd) / 300)
abs(sim.mean - rep(ltrue.c, 2)) / sim.se
square.diff1 = (ssti.results[, 1] - rep(ltrue.c, 300))^2
(1 / 300) * sum(square.diff1)
square.diff2 = (ssti.results[, 2] - rep(ltrue.c, 300))^2
(1 / 300) * sum(square.diff2)
write.csv(x = as.data.frame(ssti.results), file = "ssti001,100,1e6.csv")

colours <- c("true value" = "red")
ss.result.density = ggplot(data = as.data.frame(ssti.results[, 1]), aes(x = ssti.results[, 1])) +
  geom_density(alpha=.2, fill="lightblue") +
  xlim(-230.9, -230.6) +
  geom_vline(aes(xintercept = ltrue.c, color = "true value")) +
  labs(title = "The Stepping-Stone Estimates",
       subtitle = "v=0.01, d=100, n=1e6") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

ss.result.box = ggplot(data = as.data.frame(ssti.results[, 1]), aes(x = ssti.results[, 1])) +
  geom_boxplot() +
  xlim(-230.9, -230.6) +
  geom_vline(xintercept = ltrue.c, color = "red") +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

ss.result.density / ss.result.box +
  plot_layout(heights = c(3, 1)) +
  labs(x = "estimates of marginal likelihood",
       color = "legend")

ti.result.density = ggplot(data = as.data.frame(ssti.results[, 2]), aes(x = ssti.results[, 2])) +
  geom_density(alpha=.2, fill="lightblue") +
  xlim(-230.9, -230.6) +
  geom_vline(aes(xintercept = ltrue.c, color = "true value")) +
  labs(title = "The Thermodynamic Integration Estimates",
       subtitle = "v=0.01, d=100, n=1e6") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

ti.result.box = ggplot(data = as.data.frame(ssti.results[, 2]), aes(x = ssti.results[, 2])) +
  geom_boxplot() +
  xlim(-230.9, -230.6) +
  geom_vline(xintercept = ltrue.c, color = "red") +
  theme(axis.text.y=element_blank(),
        axis.ticks.y=element_blank())

ti.result.density / ti.result.box +
  plot_layout(heights = c(3, 1)) +
  labs(x = "estimates of marginal likelihood",
       color = "legend")