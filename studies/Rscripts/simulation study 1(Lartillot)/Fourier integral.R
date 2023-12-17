set.seed(42)
library(matrixStats)
library(mvtnorm)

inflation <- 1
iter <- 300
n = 1e3
v = 0.01
p = 10
R = 40
Rt = 2000
Re = 360
tau = exp(-7.25)
eta = exp(-7.75)
extraction_index <- 42

target.mean = rep(0, p)
target.var = diag(x = v/(v + 1), ncol = p, nrow = p)

log.true.c <- function(v, p) {
  (p / 2) * (log(v) - log(1 + v)) # this formula is given right after (58) in Lartillot.
}
ltrue.c = log.true.c(v = v, p = p)

###### delete later
likelihood = function(xi, v) 

# joint density of data and parameter
g = function(theta, v){
  like = -sum(theta^2) / (2 * v)
  prior = sum(dnorm(theta, 0, 1, log = T))
  return(like + prior)
}

epanechnikov.results = triangle.results = doubleexp.results = norm.results = simulation.results = rep(NA, iter)
simR_results = rep(NA, iter)
ptm <- proc.time()
# we choose to evaluate the posterior density at 0
for (i in 1:iter){
  if (p == 1){
    # FI
    target.sample = rnorm(n = n, sd = sqrt(target.var))
    
    a = sin(R * target.sample) / target.sample
    post.dens = abs(sum(a)) / (n * pi)
    lpriorlike = g(0, v = v)
    simulation.results[i] = lpriorlike - log(post.dens)
    
    # normal-FI
    post.dens = mean(dnorm(target.sample, mean = 0, sd = 4 * tau))
    norm.results[i] = lpriorlike - log(post.dens)
    # double-exponential-FI
    post.dens = mean(dcauchy(target.sample, location = 0, scale = eta))
    doubleexp.results[i] = lpriorlike - log(post.dens)
    # triangle-kernel-FI
    a = (1 / (Rt * target.sample^2)) * (1 - cos(Rt * (target.sample)))
    post.dens = sum(a) / (n * pi^p)
    triangle.results[i] = lpriorlike - log(post.dens)
    # Epanechnikov-kernel-FI
    a = (-2 / Re) * (1 / target.sample^2) * cos(Re * (-target.sample)) + (2 / Re^2) * (1 / -target.sample^3) * sin(Re * (-target.sample))
    post.dens = sum(a) / (n * pi^p)
    epanechnikov.results[i] = lpriorlike - log(post.dens)
  }
  if (p >= 2){
    target.sample = rnorm(p * n, mean = 0, sd = sqrt(v / (v + 1))) * inflation
    target.sample = matrix(c(target.sample), nrow = n, ncol = p, byrow = T)
    
    a = abs(rowProds(sin(R * target.sample) / target.sample)) # multiplying together different dimensions
    post.dens = sum(a) / (n * pi^p)
    lpriorlike = g(rep(0, p), v = v)
    simulation.results[i] = lpriorlike - log(post.dens)
    
    # normal-FI
    post.dens = mean(dmvnorm(target.sample, mean = rep(0, p), sigma = diag(2 * tau, nrow = p)))
    lpriorlike = g(0, v = v)
    norm.results[i] = lpriorlike - log(post.dens)
  }
  if (i == extraction_index) {
    test_sample = target.sample
  }
  print(paste0("iteration", i))
}
proc.time() - ptm
mean(is.na(simulation.results))
mean(simulation.results)
sd(simulation.results) / sqrt(iter)
square.diff = (simulation.results - rep(ltrue.c, iter))^2
(1 / iter) * sum(square.diff)

#############
# testing alternating the reference points for FI (density comparison)
#############
test_evaluation_index = sample(1:iter, iter, replace = TRUE)
if (p == 1) {
  for (j in 1:iter) {
    ref_index = test_evaluation_index[j]
    ref = test_sample[ref_index]
    evaluation_sample = test_sample[-ref_index]
    
    a = sin(R * (evaluation_sample - ref)) / (evaluation_sample - ref)
    post.dens = abs(sum(a) / (n * pi))
    lpriorlike = g(ref, v = v)
    simR_results[j] = lpriorlike - log(post.dens)
    
    print(paste0("iteration", j))
  }
} else {
  for (j in 1:iter) {
    ref_index = test_evaluation_index[j]
    ref = test_sample[ref_index, ]
    evaluation_sample = test_sample[-ref_index, ]
    
    a = abs(rowProds(sin(R * (evaluation_sample - ref)) / (evaluation_sample - ref))) # multiplying together different dimensions
    post.dens = abs(sum(a)) / (n * pi^p)
    lpriorlike = g(ref, v = v)
    simR_results[j] = lpriorlike - log(post.dens)
    
    print(paste0("iteration", j))
  }
  
}
mean(is.na(simR_results))
mean(simR_results)
sd(simR_results) / sqrt(iter)
square.diff = (simulation.results - rep(ltrue.c, iter))^2
(1 / iter) * sum(square.diff)



pdf("GFIcase4.pdf")
plot(density(simulation.results), xlab = "estimates of marginal likelihood",
     main = "The Fourier Integral Estimates")
abline(v = ltrue.c, col = "red")
mtext(substitute(paste("R = ", v), 
                 list(v = R)),
      side = 1, line = 4, col = "blue")
#legend("topright", legend = c("density of estimates", "true value"), col = c("black", "red"), lwd = 2, lty = 1)

plot(density(simR_results), xlab = "estimates of marginal likelihood",
     main = "The Fourier Integral Estimates randomising reference")
abline(v = ltrue.c, col = "red")
mtext(substitute(paste("R = ", v), 
                 list(v = R)),
      side = 1, line = 4, col = "blue")

plot(density(norm.results), xlab = "estimates of marginal likelihood",
     main = "The normal-kernel Fourier Integral Estimates")
abline(v = ltrue.c, col = "red")
mtext(substitute(paste("tau = ", v), 
                 list(v = tau)),
      side = 1, line = 4, col = "blue")

plot(density(doubleexp.results), xlab = "estimates of marginal likelihood",
     main = "The Cauchy-kernel Fourier Integral Estimates")
abline(v = ltrue.c, col = "red")
mtext(substitute(paste("eta = ", v), 
                 list(v = eta)),
      side = 1, line = 4, col = "blue")

plot(density(triangle.results), xlab = "estimates of marginal likelihood",
     main = "The triangular-kernel Fourier Integral Estimates")
abline(v = ltrue.c, col = "red")
mtext(substitute(paste("R = ", v), 
                 list(v = Rt)),
      side = 1, line = 4, col = "blue")

plot(density(epanechnikov.results), xlab = "estimates of marginal likelihood",
     main = "The Epanechnikov-kernel Fourier Integral Estimates")
abline(v = ltrue.c, col = "red")
mtext(substitute(paste("R = ", v), 
                 list(v = Re)),
      side = 1, line = 4, col = "blue")
dev.off()

mean(norm.results)
sd(norm.results) / sqrt(iter)
mean(doubleexp.results)
sd(doubleexp.results) / sqrt(iter)
mean(triangle.results)
sd(triangle.results) / sqrt(iter)
mean(epanechnikov.results)
sd(epanechnikov.results) / sqrt(iter)
