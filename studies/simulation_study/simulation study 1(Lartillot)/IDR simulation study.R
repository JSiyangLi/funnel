library(mvtnorm)
library("Rcpp")
library(ggplot2)
library(patchwork)
library(tidyverse)
set.seed(42)
logminus = function(x, y){
  if(x >= y){
    return(x + log(1.0 - exp(y - x)));
  }else{
    return(NA);
  }
}
#sourceCpp(file.choose())
sourceCpp(paste0(getwd(), "/logsum.cpp"))
single.radius.indicator <- F
radius.seq = c(0.1, 0.25, 0.5, 0.75, 1, 3)
pilot_size = 30
optimal.radius = 0.1

#####################################
### parameters of case background ###
#####################################
n = 1e6
v = 1
p = 1
target.mean = rep(0, p)
target.var = diag(x = v/(v + 1), ncol = p, nrow = p)

log.true.c <- function(v, p) {
  (p / 2) * (log(v) - log(1 + v)) # this formula is given right after (58) in Lartillot.
}
ltrue.c = log.true.c(v = v, p = p)

#########################
### functions for IDR ###
#########################
## IDR-structural functions
norm.vec = function(x){
  sqrt(sum(x^2))
}

## Recursive or direct forluma for calculating the ball volume
if (p <= 10) {
  log.ball.volume = function(p, radius){ # recursive function for low-dimensional balls
    #see wikipedia "volume of an n-ball"
    if (p == 0) {
      volume = 0
    } else if (p == 1){
      volume = log(2) + log(radius)
    } else {
      volume = log(2) + log(pi) - log(p) + 2*log(radius) + log.ball.volume(p-2, radius = radius)
    }
    return(volume)
  }
  
} else { # direct function for high-dimensional balls
  log.ball.volume = function(p, radius){
    (p / 2) * log(pi) + p * log(radius) - lgamma(p / 2 + 1)
  }
}



h = function(xi, xi.0, p, r){
  distance = norm.vec(xi - xi.0)
  output = (1 - (r^p)/(distance^p))^(1/p) * (xi - xi.0)
  return(output)
}

estimated.rmse = function(log.k, log.c.hat, n, linflated.densities, original.densities){
  # this function uses (3.21) in "IDR for marginal likelihood in Bayesian phylogenetics"
  density.ratio = exp(linflated.densities - original.densities)
  rmse = (1 / sqrt(n)) * (exp(log.c.hat - log.k)) * sd(density.ratio)
  return(rmse)
}

#################
## density functions
q = function(theta, v){
  like = -sum(theta^2) / (2 * v)
  prior = sum(dnorm(theta, 0, 1, log = T))
  return(like + prior)
}

qr = function(xi, xi.0, p, radius, v, q.top){
  # determine whether the particle is inside the unit circle from the centre (sample mean)
  distance = norm.vec(xi - xi.0)
  if (distance <= radius){ # the density value will be the maximum density (which is the origin)
    est.dens = q.top
  } 
  if (distance > radius){ 
    h.xi.diff = h(xi = xi, xi.0 = xi.0, p = p, r = radius)
    est.dens = q(theta = (xi.0 + h.xi.diff), v = v)
  }
  return(est.dens)
}

#####################
## estimation functions
c.hat.estimation.single.radius = function(p, 
                                          radius, 
                                          original.densities, 
                                          target.sample, 
                                          v, n, xi.0, q.top,
                                          single.radius.indicator){ # = the wrap.linflatedpost function
  linflated.densities = apply(X = target.sample, 1,
                              FUN = qr,p = p,
                              radius = radius,
                              xi.0 = xi.0, v = v, q.top = q.top)
  # marginal likelihood
  log.k = log.ball.volume(p = p, radius = radius) + q.top #or + q(theta = rep(0, p), v = v)
  log.dens.ratio = logplusvec(linflated.densities - original.densities)
  log.denominator = logminus(log.dens.ratio - log(n), 0)
  log.c.hat = log.k - log.denominator
  
  if (single.radius.indicator == T){
    return(log.c.hat)
  } else {
    # RMSE
    est.rmse = estimated.rmse(log.k = log.k, 
                              log.c.hat = log.c.hat, 
                              n = n, 
                              linflated.densities = linflated.densities, 
                              original.densities = original.densities)
    result.vec = c(radius, log.c.hat, est.rmse)
    names(result.vec) <- c("radius", "log c hat", "estimated RMSE")
    return(result.vec)
  }
}

c.hat.estimation.single.radius.1d = function(radius, 
                                             original.densities, 
                                             target.sample, 
                                             v, n, xi.0, q.top,
                                             single.radius.indicator){ # = the wrap.linflatedpost function
  linflated.densities = sapply(target.sample,
                               qr,p = 1,
                               radius = radius,
                               xi.0 = xi.0, v = v, q.top = q.top)
  # marginal likelihood
  log.k = log.ball.volume(p = p, radius = radius) + q.top # or + q(theta = rep(0, p), v = v)
  log.dens.ratio = logplusvec(linflated.densities - original.densities)
  log.denominator = logminus(log.dens.ratio - log(n), 0)
  log.c.hat = log.k - log.denominator
  
  if (single.radius.indicator == T){
    return(log.c.hat)
  } else {
    # RMSE
    est.rmse = estimated.rmse(log.k = log.k, 
                              log.c.hat = log.c.hat, 
                              n = n, 
                              linflated.densities = linflated.densities, 
                              original.densities = original.densities)
    result.vec = c(radius, log.c.hat, est.rmse)
    names(result.vec) <- c("radius", "log c hat", "estimated RMSE")
    return(result.vec)
  }
}

c.hat.estimation.auto = function(p, radius.seq, target.sample,original.densities,v, n, xi.0, plots = F, q.top){
  group.results = t(sapply(radius.seq,
                           c.hat.estimation.single.radius,
                           p = p, 
                           target.sample = target.sample, 
                           original.densities = original.densities, 
                           v = v, n = n, xi.0 = xi.0, q.top = q.top,
                           single.radius.indicator = F))
  min.rmse.index = which.min(group.results[, 3])
  final.estimate = group.results[min.rmse.index, ]
  if (plots == T){
    plot(density(group.results[, 2]), xlim = c(min(ltrue.c, min(group.results[, 5])), max(group.results[, 5])))
    abline(v = ltrue.c, col = "red")
    abline(v = final.estimate[2], col = "blue")
    plot(radius.seq, group.results[, 2], type = "l", ylab = "log c hat")
    abline(h = ltrue.c, col = "red")
    abline(h = final.estimate[2], col = "blue")
  }
  return(final.estimate)
}

c.hat.estimation.auto.1d = function(radius.seq, target.sample,original.densities,v, n, xi.0, plots = F, q.top){
  group.results = t(sapply(radius.seq,
                           c.hat.estimation.single.radius.1d, 
                           target.sample = target.sample, 
                           original.densities = original.densities, 
                           v = v, n = n, xi.0 = xi.0, q.top = q.top,
                           single.radius.indicator = F))
  min.rmse.index = which.min(group.results[, 3])
  final.estimate = group.results[min.rmse.index, ]
  return(final.estimate)
}

###################
### simulations ###
###################
ptm <- proc.time()
if (single.radius.indicator == T){ # only using the optimal radius for estimations
  simulation.results = rep(NA, 300)
  if (p == 1){ # this should make the algorithm run faster in the one-dimensional case
    for (i in 1:300){
      target.sample = rnorm(n = n, mean = target.mean, sd = sqrt(target.var))
      q.top = q(theta = rep(0, p), v)
      xi.0 = mean(target.sample)
      original.densities = sapply(target.sample, q, v = v)
      simulation.results[i] = c.hat.estimation.single.radius.1d(radius = optimal.radius,
                                                                original.densities = original.densities,
                                                                target.sample = target.sample,
                                                                v = v, n = n, xi.0 = xi.0, q.top = q.top,
                                                                single.radius.indicator = T)
      print(paste0("iteration", i))
    }
  }
  if (p >= 2){
    for (i in 1:300){
      target.sample = rnorm(p * n, mean = 0, sd = sqrt(v / (v + 1)))
      target.sample = matrix(c(target.sample), nrow = n, ncol = p, byrow = T)
      q.top = q(theta = rep(0, p), v)
      xi.0 = apply(target.sample, 2, mean)
      original.densities = apply(target.sample, 1, q, v = v)
      simulation.results[i] = c.hat.estimation.single.radius(p = p, radius = optimal.radius, 
                                                             original.densities = original.densities,
                                                             target.sample = target.sample,
                                                             v = v, n = n, xi.0 = xi.0, q.top = q.top,
                                                             single.radius.indicator = T)
      print(paste0("iteration", i))
    }
  }
} else { # using multiple radius for estimations and then choosing the result using estimated RMSE
  simulation.results = array(dim = c(length(radius.seq), 3, pilot_size)) # dim1:iteration; dim2: radius; dim3: outcomes
  if (p == 1){
    for (i in 1:pilot_size){
      target.sample = rnorm(n = n, mean = target.mean, sd = sqrt(target.var))
      q.top = q(theta = rep(0, p), v)
      xi.0 = mean(target.sample)
      original.densities = sapply(target.sample, q, v = v)
      simulation.results[, , i] = t(sapply(radius.seq,
                                         c.hat.estimation.single.radius.1d, 
                                         target.sample = target.sample, 
                                         original.densities = original.densities, 
                                         v = v, n = n, xi.0 = xi.0, q.top = q.top,
                                         single.radius.indicator = F))
      print(paste0("iteration", i))
    }
  }
  if (p >= 2){
    for (i in 1:pilot_size){
      target.sample = rnorm(p * n, mean = 0, sd = sqrt(v / (v + 1)))
      target.sample = matrix(c(target.sample), nrow = n, ncol = p, byrow = T)
      q.top = q(theta = rep(0, p), v)
      xi.0 = apply(target.sample, 2, mean)
      original.densities = apply(target.sample, 1, q, v = v)
      simulation.results[i, ] = c.hat.estimation.single.radius(p = p,radius.seq = radius.seq,
                                                      target.sample = target.sample,
                                                      original.densities = original.densities,
                                                      v = v, n = n, xi.0 = xi.0, q.top = q.top)
      print(paste0("iteration", i))
    }
  }
}
(proc.time() - ptm) / 300

#################################################
### post processing of the simulation results ###
#################################################
####### adjust the radiuses accordingly (depending on which case of the simulation study is being computed)
if (single.radius.indicator == T){
  (sim.mean = mean(simulation.results))
  (sim.se = sd(simulation.results) / sqrt(300))
  abs(sim.mean - ltrue.c) / sim.se
  square.diff = (simulation.results - rep(ltrue.c, 300))^2
  (1 / 300) * sum(square.diff)
  
  colours <- c("true value" = "red")
  idr.result.density = ggplot(data = as.data.frame(simulation.results), aes(x = simulation.results)) +
    geom_density(alpha=.2, fill="lightblue") +
    geom_vline(aes(xintercept = ltrue.c, color = "true value")) +
    labs(title = "The Inflated Density Ratio Estimates, radius = 0.01",
         subtitle = "d=100, v=0.01, n=1e3") +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank())
  
  idr.result.box = ggplot(data = as.data.frame(simulation.results), aes(x = simulation.results)) +
    geom_boxplot() +
    geom_vline(xintercept = ltrue.c, color = "red") +
    theme(axis.text.y=element_blank(),
          axis.ticks.y=element_blank())
  
  idr.result.density / idr.result.box +
    plot_layout(heights = c(3, 1)) +
    labs(x = "estimates of marginal likelihood",
         color = "legend")
  
} else {
  simdf = apply(simulation.results, 2, c) |> as.data.frame()
  colnames(simdf) <- c("radius", "logchat", "estimatedRMSE")
  meancol <- simdf %>%
    split(.$radius) %>%
    map_dbl(~mean(.$logchat))
  rmsecol <- simdf %>%
    split(.$radius) %>%
    map_dbl(~mean(.$estimatedRMSE))
  secol <- simdf %>%
    split(.$radius) %>%
    map_dbl(function(x) sd(x$logchat) / sqrt(pilot_size))
  cbind(meancol, rmsecol, secol)
  
  apply(simulation.results[, 2, ], 2, mean, na.rm = TRUE)
  apply(simulation.results[, 2:3, ], 2, sd) / sqrt(pilot_size)
  simulation.results[, 1]
  square.diff = (simulation.results[, 2] - rep(ltrue.c, 300))^2
  (1 / 300) * sum(square.diff)
  
  
  mean(simulation.results[, 1] == 0.1)
  which(simulation.results[, 1] == 0.1)
  plot(density(simulation.results[, 2]), xlab = "estimates of marginal likelihood",
       main = "The Inflated Density Ratio Estimates")
  abline(v = ltrue.c, col = "red")
  legend("topright", legend = c("density of estimates", "true value"), col = c("black", "red"), lwd = 2, lty = 1)
  lines(density(simulation.results[which(simulation.results[, 1] == 0.1), 2]), col = "blue")
  lines(density(simulation.results[which(simulation.results[, 1] != 0.1), 2]), col = "purple")
  legend("topright", legend = c("density of all estimates", "true value", "radius 0.1 estimates", "radius non-0.1 estimates"), 
         col = c("black", "red", "blue", "purple"), lwd = 2, lty = 1)
}
