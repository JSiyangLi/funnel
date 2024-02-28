set.seed(42)
library(matrixStats)
library(mvtnorm)
library(ggfortify)
library(ggplot2)



fi_sim <- function(p)
{
  


inflation <- 1
iter <- 300
n = 1e3
v = 0.01
#p = 10
R = 400
Rt = 2000
Re = 360
tau = exp(-7.25)
eta = exp(-7.75)
extraction_index <- 4

target.mean = rep(0, p)
target.var = diag(x = v/(v + 1), ncol = p, nrow = p)

log.true.c <- function(v, p) {
  (p / 2) * (log(v) - log(1 + v)) # this formula is given right after (58) in Lartillot.
}
ltrue.c = log.true.c(v = v, p = p)

###### delete later
# likelihood = function(xi, v)

# joint density of data and parameter
likepri_fn <- function(theta, v){
  like = -sum(theta^2) / (2 * v)
  prior = sum(dnorm(theta, 0, 1, log = T))
  return(like + prior)
}

epanechnikov.results = triangle.results = doubleexp.results = norm.results = simulation.results = rep(NA, iter)
simR_results = simR_lpriorlike = rep(NA, iter)
ptm <- proc.time()
# we choose to evaluate the posterior density at 0
for (i in 1:iter){
  if (p == 1){
    # FI
    target.sample = rnorm(n = n, sd = sqrt(target.var))
    
    a = sin(R * target.sample ) / target.sample
    post.dens = abs(sum(a)) / (n * pi)
    lpriorlike = likepri_fn(0, v = v)
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
    lpriorlike = likepri_fn(rep(0, p), v = v)
    simulation.results[i] = lpriorlike - log(post.dens)
    
    # normal-FI
    post.dens = mean(dmvnorm(target.sample, mean = rep(0, p), sigma = diag(2 * tau, nrow = p)))
    lpriorlike = likepri_fn(0, v = v)
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
    simR_lpriorlike[j] = likepri_fn(ref, v = v)
    simR_results[j] = simR_lpriorlike[j] - log(post.dens)
    
    print(paste0("iteration", j))
  }
} else {
  for (j in 1:iter) {
    ref_index = test_evaluation_index[j]
    ref = test_sample[ref_index, ]
    evaluation_sample = test_sample[-ref_index, ]
    
    a = abs(rowProds(sin(R * (evaluation_sample - ref)) / (evaluation_sample - ref))) # multiplying together different dimensions
    post.dens = abs(sum(a)) / (n * pi^p)
    simR_lpriorlike[j] = likepri_fn(ref, v = v)
    simR_results[j] = simR_lpriorlike[j] - log(post.dens)
    
    print(paste0("iteration", j))
  }
  
}
mean(is.na(simR_results))
mean(simR_results)
sd(simR_results) / sqrt(iter)
square.diff = (simulation.results - rep(ltrue.c, iter))^2
(1 / iter) * sum(square.diff)

# OLS and PCA
simR_fit <- lm(simR_results ~ simR_lpriorlike)
simR_fit |> summary()
simR_pc <- prcomp(cbind(simR_lpriorlike, simR_results))

# derive the rightmost cluster variance using the rightmost r% points
left_bound <- sort(simR_lpriorlike, decreasing = TRUE)[ceiling(0.05 * length(simR_lpriorlike))]
rightmost_var <- var(simR_results[which(simR_lpriorlike >= left_bound)])

# derive the 'angular' bias estimate
simR_rotation <- min(abs(acos(simR_pc$rotation[1, 1]) - pi), acos(simR_pc$rotation[1, 1]))

# loss function
simR_loss <- function(rotation_bias, cluster_variance) {
  exp(2 * rotation_bias) + cluster_variance
}
simR_loss(simR_rotation, rightmost_var)

####### plotting

upper = max(c(max(simulation.results), ltrue.c)) + 10
lower = min(c(min(simulation.results), ltrue.c)) - 10
label =  sprintf("R=%d, p=%d, n=%d", R, p, n)
png(file=sprintf("%s.png", label))
plot(density(simulation.results), xlab = "estimates of marginal likelihood",
     main = "The Fourier Integral Estimates",
     xlim=c(lower,upper)
     )
abline(v = ltrue.c, col = "red")
mtext(label, side = 1, line = 4, col = "blue")
dev.off()

}



fi_sim(1)
fi_sim(20)
fi_sim(100)
