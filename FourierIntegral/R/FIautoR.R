FIautoR <-
function(# basic functionality arguments
                    evaluation_pt = NULL,
                    use_kernel_value = FALSE, # use kernel value or kernel function for the computation?
                    posterior_kernel = NULL,
                    posterior_sample,
                    kernel_value = NULL, # enabling the function to compute FI without knowing the posterior kernel function
                    # standardising the posterior sample?
                    standardisation = TRUE,
                    # R candidates creation
                    Rvec = seq(from = 10, to = 2500, by = 10),
                    # R selection
                    Rchangepoint = TRUE, # use the selection criterion based on change point or median?
                    Rchangepoint_continuous = FALSE, # use a continuous prior for the change point parameter?
                    Rone_plus_sd = TRUE, # use venilla posterior mean or one plus sd of the posterior means of the change point?
                    # R change-point JAGS arguments
                    burn_in = 1000, # How many burn-in steps?
                    steps = 5000, # How many proper steps?
                    thin = 1, # Thinning?
                    # R median arguments
                    Rtolerance = 0.5, # how far from the median do we allow the candidates to deviate?
                    Rmindist = FALSE, # use the minimum-distance-to-median or in-the-tube criterion?
                    # function output arguments
                    message_suppress = FALSE, # print messages?
                    plotting = FALSE, # plot the estimates of different R values? (with all results listed)
                    mar = c(5.1, 4.1, 4.1, 2.1), # plot of FI estimates margins
                    ...) {
  # input validity
  #if (!is.data.frame(posterior_sample))
  #  stop("The posterior sample needs to be in a data frame")
  if (!is.numeric(Rvec))
    stop("R needs to be numeric")
  if (!is.function(posterior_kernel) && !is.numeric(kernel_value))
    stop("At least one of posterior kernel function or kernel values of the posterior sample should be supplied")
  
  if (!standardisation) { # not standardising the posterior sample
    # evaluating posterior density
    if (!use_kernel_value) { # use the posterior kernel function
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
      
      lq <- posterior_kernel(..., xi = evaluation_pt)
    } else { # use posterior kernel value
      evaluation_pt_index <- sample(1:nrow(posterior_sample), size = 1, prob = kernel_value)
      lq <- kernel_value[evaluation_pt_index]
      if (is.null(dim(posterior_sample))) { # the parameter space is univariate
        posterior_sample <- posterior_sample[-evaluation_pt_index]
      } else {
        posterior_sample <- posterior_sample[-evaluation_pt_index, ] # remove the evaluation point from the posterior sample
      }
    }
    
    # FI multiple R computation
    if (is.null(dim(posterior_sample))) { # the parameter space is univariate
      n <- length(posterior_sample)
      post_diff <- posterior_sample - evaluation_pt
      estimates <- sapply(Rvec, function(r) {
        a <- sin(r * post_diff) / post_diff
        lpostdens <- log(abs(sum(a))) - (log(n) + log(pi)) # Fourier integral theorem
        lq - lpostdens
      })
    } else { # higher-dim parameter space
      n <- nrow(posterior_sample)
      x_mat <- matrix(rep(evaluation_pt, n), nrow = n, byrow = TRUE)
      post_diff <- as.matrix(posterior_sample - x_mat)
      estimates <- sapply(Rvec, function(r) {
        a <- rowProds(sin(r * post_diff) / post_diff)
        lpostdens <- log(abs(sum(a))) - (log(n) + ncol(posterior_sample) * log(pi)) # Fourier integral theorem
        lq - lpostdens
      })
    }
  } else { # standardising the posterior sample
    # evaluating posterior density
    if (!use_kernel_value) { # use the posterior kernel function
      # standardization
      if (is.null(dim(posterior_sample))) { # univariate
        post_means <- mean(posterior_sample)
        post_sds <- sd(posterior_sample)
        posterior_sample <- (posterior_sample - post_means) / post_sds
      } else { # multivariate
        posterior_sample.mat <- as.matrix(posterior_sample)
        post_means <- colMeans(posterior_sample.mat)
        post_sds <- colSds(posterior_sample.mat)
        posterior_sample <- sweep(posterior_sample.mat, MARGIN = 2, STATS = post_means) |>
          sweep(MARGIN = 2, STATS = post_sds, FUN = "/") |> 
          as.data.frame()
      }
      
      # setting default evaluation point to the posterior mean
      if (!is.numeric(evaluation_pt)) {
        if (!message_suppress) message("Evaluation point not provided or invalid, using posterior mean (0)")
        
        evaluation_pt <- if (is.null(dim(posterior_sample))) {0} else {rep(0, ncol(posterior_sample))}
        lq <- posterior_kernel(..., xi = evaluation_pt) # evaluating at the posterior mean
      } else { # transformed evaluation point is specified
        lq <- posterior_kernel(..., xi = evaluation_pt)
        evaluation_pt <- (evaluation_pt - post_means) / post_sds
      }
      
    } else { # use posterior kernel value
      evaluation_pt_index <- sample(1:nrow(posterior_sample), size = 1, prob = kernel_value)
      lq <- kernel_value[evaluation_pt_index]
      if (is.null(dim(posterior_sample))) { # the parameter space is univariate
        posterior_sample <- posterior_sample[-evaluation_pt_index]
      } else {
        posterior_sample <- posterior_sample[-evaluation_pt_index, ] # remove the evaluation point from the posterior sample
      }
    }
    
    # FI multiple R computation
    if (is.null(dim(posterior_sample))) { # the parameter space is univariate
      n <- length(posterior_sample)
      post_diff <- posterior_sample - evaluation_pt
      estimates <- sapply(Rvec, function(r) {
        a <- sin(r * post_diff) / post_diff
        lpostdens <- log(abs(sum(a))) - (log(n) + log(pi)) # Fourier integral theorem
        lq - lpostdens + log(post_sds)
      })
    } else { # higher-dim parameter space
      n <- nrow(posterior_sample)
      x_mat <- matrix(rep(evaluation_pt, n), nrow = n, byrow = TRUE)
      post_diff <- as.matrix(posterior_sample - x_mat)
      estimates <- sapply(Rvec, function(r) {
        a <- rowProds(sin(r * post_diff) / post_diff)
        lpostdens <- log(abs(sum(a))) - (log(n) + ncol(posterior_sample) * log(pi)) # Fourier integral theorem
        lq - lpostdens + sum(log(post_sds))
      })
    }
  }
  
  # removing NaN
  estR <- Rvec[which(!is.na(estimates))]
  estValues <- estimates[which(!is.na(estimates))]
  
  # R selection
  if (Rchangepoint) {
    kcont <- Rchangepoint_continuous
    if (Rchangepoint_continuous) {
      BUGSmodel = "model
      { # prior
        beta ~ dnorm(0, 0.001)
        delta ~ dnorm(0.001,0.001)
  
        mu2 ~ dnorm(0, 0.001)
        tau2 ~ dnorm(0.001,0.001)
        k ~ dunif(min(R), max(R)) # location of the change point
  
        # likelihood
        for (i in 1:Rlen){
          J[i] <- step(R[i] - k)
          mu[i] <- beta * R[i] + J[i] * mu2
          logtau[i] <- delta * R[i] + J[i] * tau2
          tau[i] <- exp(logtau[i])
          chat[i] ~ dnorm(mu[i],tau[i])
        }
      }
      "
      Rlen <- length(estValues)
      # The data (use NA for no data) as a list
      data = list(chat = estValues - mean(estValues), Rlen = Rlen, R = estR - mean(estR))
      # The initial values as a list (JAGS will automatically generate these if unspecified)
      inits = list(beta = 1, mu2 = 1, tau2 = 1, k = 50)
      # parameters to monitor
      parameters = c('beta', 'mu2', 'delta', 'tau2', 'k')
      #compilation of the BUGS model, no Gibbs sampling yet
      foo <- jags.model(textConnection(BUGSmodel),data=data, inits=inits, n.chains=1)
      #burnin samples
      update(foo, burn_in)
      #draws n.iter MCMC samples, monitors the parameters specified in variable.names, thins the
      #output by a factor of thin and stores everything in a MCMC.list object
      out <- coda.samples(model=foo, variable.names=parameters, n.iter=steps, thin=thin, n.chains=1)
      
      # the estimated change point with a continuous prior may sometimes be invalid:
      if (mean(out[[1]][, "k"]) >= min(estR) && mean(out[[1]][, "k"]) <= max(estR)) { # the result is valid
        index <- which(estR - mean(out[[1]][, "k"]) > 0)[1]
        output <- estValues[index]
        kcont <- TRUE # indicator for if k is continuous
        
        # sanity check: whether change point model is
        if (which(estR - median(out[[1]][, "k"]) > 0)[1] == 1) {
          warning("The estimates may look like a horizontal band.")
        } else if (index / Rlen > 0.75 || Rlen - index < 30) {
          warning("Too few computed R values. Flat tail check may be unreliable.")
        }
        
        # weighted average of estimates based on posterior probs of change points
        if (length(Rvec) != Rlen) warning("some estimates are invalid, posterior predictive estimate unreliable.")
        post_vec <- tabulate(out[[1]][, "k"] + 1, nbins = max(estR))[-(1:min(estR))]
        post_freq <- c(0, rowSums(matrix(post_vec, ncol = diff(Rvec)[1], byrow = TRUE))) #diff(Rvec)[1]=Rby
        postpred_estimate = as.numeric((post_freq / sum(post_freq)) %*% estValues)
        
      } else { # the result is invalid, use discrete prior on k
        if (!message_suppress) message("Continuous change-point prior failed, using discrete change-point prior.")
        kcont <- FALSE
      }
    } 
    
    if (!kcont || !Rchangepoint_continuous) {
      kcont <- FALSE
      
      # discrete-prior k
      BUGSmodel = "model
      { # prior
        beta ~ dnorm(0, 0.001)
        delta ~ dnorm(0, 0.001) # as the y-axis is shifted to the change point, (R[i]-R[k]) is negative.
  
        mu2 ~ dnorm(0, 0.001)
        tau2 ~ dnorm(0, 0.001)
        k ~ dcat(p) # index of the position of the change point
  
        # likelihood
        for (i in 1:Rlen){
          J[i] <- step(i-k+0.5) #+0.5 due to the case the 'change point' is before the first observation
          mu[i] <- (1 - J[i]) * (beta * (R[k] - R[i]) + mu2) + J[i] * mu2
          logtau[i] <- (1 - J[i]) * delta + J[i] * tau2
          tau[i] <- exp(logtau[i])
          chat[i] ~ dnorm(mu[i], tau[i])
        }
      }
      "
      Rlen <- length(estValues)
      p <- rep(1/Rlen, Rlen)
      # The data (use NA for no data) as a list
      data = list(chat = estValues - mean(estValues), p = p, Rlen = Rlen, R = estR - mean(estR))
      # The initial values as a list (JAGS will automatically generate these if unspecified)
      inits = list(beta = 1, mu2 = 1, tau2 = 1, k = 5)
      # parameters to monitor
      parameters = c('beta', 'mu2', 'delta', 'tau2', 'k')
      #compilation of the BUGS model, no Gibbs sampling yet
      foo <- jags.model(textConnection(BUGSmodel),data=data, inits=inits, n.chains=1)
      #burnin samples
      update(foo, burn_in)
      #draws n.iter MCMC samples, monitors the parameters specified in variable.names, thins the
      #output by a factor of thin and stores everything in a MCMC.list object
      out <- coda.samples(model=foo, variable.names=parameters, n.iter=steps, thin=thin, n.chains=1)
      
      index <- ifelse(Rone_plus_sd, # 1+SD indicator
                      ceiling(mean(out[[1]][, "k"]) - 1), # venilla posterior mean change point
                      ceiling(mean(out[[1]][, "k"]) + sd(out[[1]][, "k"]) - 1)) # 1+SD or change point
      output <- estValues[index]
      
      # sanity check: whether change point model is
      if (ceiling(median(out[[1]][, "k"] - 1)) == 1) {
        warning("The estimates may look like a horizontal band.")
      } else if (index / Rlen > 0.75 || Rlen - index < 30) {
        warning("Too few computed R values. Flat tail check may be unreliable.")
      }
      
      if (index <= 3) warning("Too few estimates before the estimated change point, estimated trend and variability will be poor.")
      
      # weighted average of estimates based on posterior probs of change points
      post_freq <- tabulate(out[[1]][, "k"], Rlen)
      postpred_estimate = as.numeric((post_freq / sum(post_freq)) %*% estValues)
    }
    
    
  
  } else {
    # R selection based on median
    distances <- abs(estimates - median(estimates, na.rm = TRUE))
    index <- if (Rmindist) { # use the 'minimum-distance-to-median' criteria.
      which.min(distances)
    } else { # use the first candidate in the tube.
      min(which(distances <= Rtolerance))
    }
    output <- estimates[index]
  }
  
  # flat tail check (OLS)
  if (Rchangepoint) {
    fit <- lm(estValues[index:Rlen] ~ estR[index:Rlen])
  } else {
    fit <- lm(estimates[index:length(Rvec)] ~ Rvec[index:length(Rvec)])
  }
  
  if (plotting) {
    layout(1)
    par(mar = mar)
    plot(Rvec, estimates, pch = 20, pty = "b", main = "FI estimates for each R")
    abline(h = median(estimates[Rvec >= 100], na.rm = TRUE), col = "green4")
    grid()
    if (Rchangepoint) { # change-point-model based selection of R
      abline(h = postpred_estimate, col = "orange")
      abline(h = output, col = "navy")
      if (kcont) { # k is continuous
        abline(v = mean(out[[1]][, "k"]), col = "blue")
        abline(v = mean(out[[1]][, "k"]) - sd(out[[1]][, "k"]), col = "blue", lty = "dashed")
        abline(v = mean(out[[1]][, "k"]) + sd(out[[1]][, "k"]), col = "blue", lty = "dashed")
        abline(h = estValues[which(estR - mean(out[[1]][, "k"]) + sd(out[[1]][, "k"]) > 0)[1]], col = "navy", lty = "dashed")
        abline(h = estValues[which(estR - mean(out[[1]][, "k"]) - sd(out[[1]][, "k"]) > 0)[1]], col = "navy", lty = "dashed")
      } else { # k is discrete
        abline(v = estR[ceiling(mean(out[[1]][, "k"]) - 1)], col = "blue")
        abline(v = estR[ceiling(mean(out[[1]][, "k"]) - sd(out[[1]][, "k"]) - 1)], col = "blue", lty = "dashed")
        abline(v = estR[ceiling(mean(out[[1]][, "k"]) + sd(out[[1]][, "k"]) - 1)], col = "blue", lty = "dashed")
        abline(h = estValues[ceiling(mean(out[[1]][, "k"]) + sd(out[[1]][, "k"]) - 1)], col = "navy", lty = "dashed")
        abline(h = estValues[ceiling(mean(out[[1]][, "k"]) - sd(out[[1]][, "k"]) - 1)], col = "navy", lty = "dashed")
      }
      legend("topright", legend = c("median", "change point R", "change point estimate", "posterior predictive estimate"), col = c("green4", "blue", "navy", "orange"), lwd = 1)
    } else { # median based selection of R
      legend("topright", legend = "median", col = "green4", lwd = 1)
    }
    mtext(substitute(paste("invalid candidates: ", v, "; total candidates: ", w), 
                     list(v = sum(is.na(estimates)), w = length(Rvec))),
          side = 1, line = 4, col = "red")
    
    if (Rchangepoint) { # change-point JAGS output
      #Obtain traceplots and kernel density estimates for each parameter
      par(mar = rep(2, 4)) #adjusts margin plotting parameters
      plot(out)
      autocorr.plot(out)
      invisible(list("Rcandidates" = Rvec, "estimates" = estimates,
                     "result" = output, "selected_R" = Rvec[index],
                     "posterior_predictive_estimate" = postpred_estimate,
                     "JAGSsummary" = summary(out), "JAGSRaftery" = raftery.diag(out)
                     #"JAGSGeweke" = geweke.diag(out)
                     ))
    } else { # median output
      invisible(list("Rcandidates" = Rvec, "result" = estimates))
    }
  } else { # no detailed results
    ifelse(Rchangepoint, list("result" = output, "selected_R" = estR[index]), list("result" = output, "selected_R" = Rvec[index]))
  }
}
