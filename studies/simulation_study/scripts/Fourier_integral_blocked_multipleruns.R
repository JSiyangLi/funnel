set.seed(42)
library(matrixStats)
library(mvtnorm)

n <- 1e4
R <- 1e10
FourierIntegral_blocked <- function(ppb, block, R) {
  # ppb = p per block
  
  # true underlying values
  log.true.c <- function(v, p) {
    (ppb * block / 2) * (log(v) - log(1 + v)) # this formula is given right after (58) in Lartillot.
  }
  ltrue.c = log.true.c(v = v, p = p)
  ltrue.density = dmvnorm(x = rep(0, ppb * block), sigma = diag(x = v, ncol = ppb * block, nrow = ppb * block), log = TRUE)
  
  # unnormalised posterior density
  g = function(theta, v){
    like = -sum(theta^2) / (2 * v)
    prior = sum(dnorm(theta, 0, 1, log = T))
    return(like + prior)
  }
  
  # initialisation
  density.results = simulation.results = rep(NA, 300)
  
  # FI
  ptm <- proc.time()
  # we choose to evaluate the posterior density at 0
  for (i in 1:300){
    post.dens_i <- rep(NA, block)
    for (block_i in 1:block) {
      # generate posterior sample for block i (independence)
      target.sample = rnorm(ppb * n, mean = 0, sd = sqrt(v / (v + 1)))
      target.sample = matrix(c(target.sample), nrow = n, ncol = ppb, byrow = T)
      
      # density estimation for block i
      as = rowProds(sin(R * target.sample) / target.sample)
      post.dens_i[block_i] = sum(as) / (n * pi^ppb)
    }
    # density estimation altogether
    post.dens = sum(log(abs(post.dens_i)))
    density.results[i] = post.dens
    
    lpriorlike = g(rep(0, ppb), v = v)
    simulation.results[i] = lpriorlike - post.dens #+ sum(log(target.samplesd))
  }
  
  # output
  c(mean(simulation.results), ltrue.c, mean(simulation.results) - ltrue.c,
    mean(density.results), ltrue.density, mean(density.results) - ltrue.density)
}

#######################
# fixing dimension per block, changing number of blocks
#########################
ppb = 1
dim_eval <- c(1, 3, 5, 10, 15, 20, 30, 50, 75, 100)
FIblocks <- sapply(dim_eval, FourierIntegral_blocked, ppb = ppb, R = R)
plot(dim_eval, FIblocks[3, ], type = "l", xlab = "total dimension (number of blocks)", 
     ylab = "evidence estimate biases", main = "Fourier Integral biases")
mtext(substitute(paste("R = ", v), 
                 list(v = R)),
      side = 1, line = 4, col = "blue")
plot(dim_eval, FIblocks[6, ], type = "l", xlab = "total dimension (number of blocks)",
     ylab = "density estimate biases")
mtext(substitute(paste("dimension per block = ", v), 
                 list(v = ppb)),
      side = 1, line = 4, col = "blue")


#########################
# fixing total dimension, changing number of blocks
#########################
#total_dim <- 100
#ppb_eval <- c(1, 2, 5, 10, 20, 25, 50, 100)

#total_dim <- 20
#ppb_eval <- c(1, 2, 4, 5, 10, 20)
# or
total_dim <- 10
ppb_eval <- c(1, 2, 5, 10)

n_blocks <- total_dim %/% ppb_eval
FI_fixdim <- sapply(ppb_eval, function(ppb) FourierIntegral_blocked(ppb = ppb, block = total_dim %/% ppb, R = R))
plot(n_blocks, FI_fixdim[3, ], type = "l", xlab = "number of blocks", 
     ylab = "log evidence estimate biases", main = "Fourier Integral biases")
abline(h = 0, col = 'red')
mtext(substitute(paste("R = ", v), 
                 list(v = R)),
      side = 1, line = 4, col = "blue")
plot(ppb_eval, FI_fixdim[6, ], type = "l", xlab = "number of blocks",
     ylab = "log density estimate biases")
abline(h = 0, col = 'red')
mtext(substitute(paste("total dimension = ", v), 
                 list(v = total_dim)),
      side = 1, line = 4, col = "blue")


#######################
# R vs log evidence estimate (different blocks overlapped)
########################
# setting R candidates
Rvec <- c(10, 25, 50, 75, 100, 200, 500, 1e3, 1e4, 1e7, 1e10, 1e15)
FourierIntegral_blocked_multipleR <- 
  function(ppb, block, Rvec)
    sapply(Rvec, FourierIntegral_blocked, ppb = ppb, block = block)

# setting blocks 
total_dim <- 100
ppb_eval <- c(1, 5, 10, 25, 100)
n_blocks <- total_dim %/% ppb_eval

# run and plot
FI_overlap_blocks <- sapply(ppb_eval, 
                            function(ppb) FourierIntegral_blocked_multipleR(ppb = ppb, block = total_dim %/% ppb, Rvec = Rvec))

FI_overlaps_evidence <- FI_overlap_blocks[c(3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69), ]
FI_overlaps_density <- FI_overlap_blocks[c(6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72), ]
#notrun
par(las = 1)
plot(log10(Rvec), log10(FI_overlaps_evidence[, 1]), type = "l", xlab = "log10(R)", 
     ylab = "log10(evidence estimate biases)", main = "Fourier Integral evidence biases", ylim = c(0, 2.2))
lines(log10(Rvec), log10(FI_overlaps_evidence[, 2]), col = rgb(0, 0.1, 0.9))
lines(log10(Rvec), log10(FI_overlaps_evidence[, 3]), col = rgb(0, 0.4, 0.6))
lines(log10(Rvec), log10(FI_overlaps_evidence[, 4]), col = rgb(0, 0.6, 0.4))
lines(log10(Rvec), log10(abs(FI_overlaps_evidence[, 5])), col = rgb(0, 0.9, 0.1))
abline(h = 0, col = "red")
mtext(substitute(paste("total dimension = ", v), 
                 list(v = total_dim)),
      side = 1, line = 4, col = "blue")
mtext("No. blocks = 1 has bias negated", side = 3, col = rgb(0, 0.9, 0.1))
legend("bottomleft", legend = n_blocks,
       col = c("black", rgb(0, 0.1, 0.9), rgb(0, 0.4, 0.6), rgb(0, 0.6, 0.4), rgb(0, 0.9, 0.1)), lty = 1)
text(2, 0.75, "No. blocks")

plot(log10(Rvec), log10(abs(FI_overlaps_density[, 1])), type = "l", xlab = "log10(R)", 
     ylab = "log10(density estimate biases)", main = "Fourier Integral density biases", ylim = c(0, 2.2))
lines(log10(Rvec), log10(FI_overlaps_density[, 2]), col = rgb(0, 0.1, 0.9))
lines(log10(Rvec), log10(FI_overlaps_density[, 3]), col = rgb(0, 0.4, 0.6))
lines(log10(Rvec), log10(FI_overlaps_density[, 4]), col = rgb(0, 0.6, 0.4))
lines(log10(Rvec), log10(abs(FI_overlaps_density[, 5])), col = rgb(0, 0.9, 0.1))
mtext(substitute(paste("total dimension = ", v), 
                 list(v = total_dim)),
      side = 1, line = 4, col = "blue")
mtext("No. blocks = 100 has bias partly negated", side = 3)
legend("bottomleft", legend = n_blocks,
       col = c("black", rgb(0, 0.1, 0.9), rgb(0, 0.4, 0.6), rgb(0, 0.6, 0.4), rgb(0, 0.9, 0.1)), lty = 1)
text(2, 0.75, "blocks")

##########################
# histograms / density plots for dimensions = 20, 100 (dim=1 is not interesting for blocking)
#########################

