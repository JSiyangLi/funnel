FI <-
function(evaluation_pt = NULL, # point of evaluation
               posterior_kernel, # user-specified posterior kernel function
               posterior_sample, #posterior sample
               R = 50, # limiting factor
               message_suppress = FALSE, # print messages?
               ... # all the arguments needed for the posterior kernel function
               ) {
  # input validity
  if (!is.data.frame(posterior_sample))
    stop("The posterior sample needs to be in a data frame")
  if (!is.numeric(R))
    stop("The limiting factor needs to be numeric")
  if (!is.function(posterior_kernel))
    stop("Invalid posterior kernel function")
  
  # setting default evaluation point to the posterior mean
  if (!is.numeric(evaluation_pt)) {
    if (!message_suppress) message("Evaluation point not provided or invalid, using posterior mean")
    
    evaluation_pt <- # ifelse() can not be used as it only returns the first value of the mean vector (-110)
      if (is.null(dim(posterior_sample))) { # the parameter space is univariate
        mean(posterior_sample)
      } else { # the parameter space is multivariate
        colMeans(posterior_sample)
      }
  }
  
  # FI computation
  n <- nrow(posterior_sample)
  lq <- posterior_kernel(..., xi = evaluation_pt)
  if (is.null(n)) { # the parameter space is univariate
    post_diff <- posterior_sample - evaluation_pt
    a <- sin(R * post_diff) / post_diff
    lpostdens <- log(sum(a)) - (log(n) + log(pi)) # Fourier integral theorem
  } else { # higher-dim parameter space
    x_mat <- matrix(rep(evaluation_pt, n), nrow = n, byrow = TRUE)
    post_diff <- as.matrix(posterior_sample - x_mat)
    a <- rowProds(sin(R * post_diff) / post_diff)
    lpostdens <- log(sum(a)) - (log(n) + ncol(posterior_sample) * log(pi)) # Fourier integral theorem
  }
  lq - lpostdens
}
