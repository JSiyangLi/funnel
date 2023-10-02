\name{FIautoR}
\alias{FIautoR}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Marginal likelihood estimate using Fourier integral method
}
\description{
Computing marginal likelihood based on Chib's method while using Fourier integral theorem for density estimation.
}
\usage{
FIautoR(evaluation_pt = NULL, use_kernel_value = FALSE, posterior_kernel, posterior_sample, kernel_value, standardisation = TRUE, Rvec = seq(from = 10, to = 2500, by = 10), Rchangepoint = TRUE, Rchangepoint_continuous = FALSE, Rone_plus_sd = TRUE, burn_in = 1000, steps = 5000, thin = 1, Rtolerance = 0.5, Rmindist = FALSE, message_suppress = FALSE, plotting = FALSE, mar = c(5.1, 4.1, 4.1, 2.1), ...)

FI(evaluation_pt = NULL, posterior_kernel, posterior_sample, R = 50, message_suppress = FALSE, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{evaluation_pt}{
 A point in the parameter space used as the evaluation (reference) point for Fourier integral method. If no point is supplied, then the posterior mean will be used.
}
  \item{use_kernel_value}{
 Logical. If \verb|TRUE| then a vector of posterior kernel values should be supplied to be used directly for the computation. Otherwise, the posterior kernel function should be supplied to be evaluated at the reference point. Default is \verb|FALSE| (i.e. posterior kernel function used).
}
  \item{posterior_kernel}{
 If \verb|use_kernel_value| is \verb|FALSE|, then a posterior kernel function should be supplied via this argument.
}
  \item{posterior_sample}{
 The posterior sample that the computation will be based on.
}
  \item{kernel_value}{
  If \verb|use_kernel_value| is \verb|TRUE|, then the posterior kernel values of the entire posterior sample should be supplied via this argument.
}
  \item{standardisation}{
 Logical. If \verb|TRUE|, then the function standardises the posterior sample before computing the marginal likelihood. The default is \verb|TRUE|.
}
  \item{Rvec}{
 A vector of all R candidates to be considered for computation and selection. The default is \verb|seq(10, 10, 2500)|.
}
  \item{Rchangepoint}{
 Logical. If \verb|TRUE|, then a change point model is used to for \verb|R| selection. Otherwise, median is used. The default is \verb|TRUE|.
}
  \item{Rchangepoint_continuous}{
 Logical. If \verb|TRUE|, then the change point is treated as a continuous parameter in the Gibbs sampler, leading to a different model specification than a discrete-parameter change-point model. The default is \verb|FALSE|.
}
  \item{Rone_plus_sd}{
 Logical. If \verb|FALSE|, then the optimal \verb|R| is selected solely based on the posterior mean of the change point parameter. Otherwise, the optimal \verb|R| is selected based on the posterior mean plus one standard deviation of the change point parameter. The default is \verb|TRUE|.
}
  \item{burn_in}{
 Numeric. The number of burn-in iterations for the Gibbs sampler to run the change point model. The default is 1000.
}
  \item{steps}{
 Numeric. The number of proper iterations for the Gibbs sampler to run the change point model. The default is 5000.
}
  \item{thin}{
 Numeric. The thinning parameter for the Gibbs sampler to run the change point model. The default is 1.
}
  \item{Rmindist}{
 Logical. If \verb|TRUE|, then the estimate closest to the median is selected. Otherwise, the first estimate inside the tolerance region is selected. The default is \verb|FALSE|.
}
  \item{Rtolerance}{
 Numeric. If \verb|Rchangepoint| is \verb|FALSE| (i.e. \verb|R| selection is based on median), then this number specifies how far away from the median should be the tolerance region, and hence the first estimated result within the tolerance region is selected. The default is 0.5.
}
  \item{message_suppress}{
 Logical. If \verb|TRUE|, then the function messages are suppressed. The default is \verb|FALSE|.
}
  \item{plotting}{
  Logical. This parameter alters the function output. If \verb|TRUE|, then the detailed results of the function is returned, including all \verb|R| candidates and their corresponding FI estimates and their plot, change point model Gibbs sampler convergence diagnostics (if a change point model is used to determine the optimal \verb|R|) and their plots. Otherwise, only the selected \verb|R| and estimate will be returned.
}
  \item{mar}{
 Margins of the estimates-per-\verb|R| plot if \verb|plotting| is \verb|TRUE|.
}
  \item{\dots}{
 Further arguments to be passed on to the posterior kernel function.
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
For mathematical details of the FI method, see Ho & Walker (2021) and Rotiroti & Walker (2022).

Raftery & Lewis diagnostics test from the \verb|coda| package has been implemented if the change point model is used for \verb|R| selection. Geweke's convergence diagnostic is also suitable for convergence diagnostics. However, there are some bugs in \verb|geweke.diag()| in the \verb|coda| package that will make the function run into an error.
}
\value{
 If \verb|plotting| is \verb|TRUE|, then the following items are returned:
  \item{result} {The Fourier integral marginal likelihood estimate based on the optimal \verb|R|.}
  \item{selected_R} {The optimal \verb|R| selected.}
 If \verb|plotting| is \verb|FALSE|, and if the change point model is used, then the following values are invisibly returned in addition to the values above:
  \item{posterior_predictive_estimate} {A posterior predictive estimate of the change point parameter.}
  \item{JAGSsummary} {The JAGS summary output of the change point model.}
  \item{JAGSRaftery} {Raftery & Lewis diagnostic test of MCMC chain convergence.}
  
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
Siyang Li
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (evaluation_pt = NULL, use_kernel_value = FALSE, posterior_kernel, 
    posterior_sample, kernel_value, standardisation = TRUE, Rvec = seq(from = 10, to = 2500, by = 10), Rchangepoint = TRUE, Rchangepoint_continuous = FALSE, 
    Rone_plus_sd = TRUE, burn_in = 1000, steps = 5000, thin = 1, 
    Rtolerance = 0.5, Rmindist = FALSE, message_suppress = FALSE, 
    plotting = FALSE, mar = c(5.1, 4.1, 4.1, 2.1), ...) 
{
    if (!is.data.frame(posterior_sample)) 
        stop("The posterior sample needs to be in a data frame")
    if (!is.numeric(Rby) || !is.numeric(Rmax)) 
        stop("The limiting factor quantities needs to be numeric")
    if (!is.function(posterior_kernel) && !is.numeric(kernel_value)) 
        stop("At least one of posterior kernel function or kernel values of the posterior sample should be supplied")
    if (!standardisation) {
        if (!use_kernel_value) {
            if (!is.numeric(evaluation_pt)) {
                if (!message_suppress) 
                  message("Evaluation point not provided or invalid, using posterior mean")
                evaluation_pt <- if (is.null(dim(posterior_sample))) {
                  mean(posterior_sample)
                }
                else {
                  colMeans(posterior_sample)
                }
            }
            Rvec <- seq(Rmin, Rmax, Rby)
            n <- nrow(posterior_sample)
            lq <- posterior_kernel(..., xi = evaluation_pt)
        }
        else {
            evaluation_pt_index <- sample(1:nrow(posterior_sample), 
                size = 1, prob = kernel_value)
            lq <- kernel_value[evaluation_pt_index]
            if (is.null(dim(posterior_sample))) {
                posterior_sample <- posterior_sample[-evaluation_pt_index]
            }
            else {
                posterior_sample <- posterior_sample[-evaluation_pt_index, 
                  ]
            }
        }
        if (is.null(n)) {
            post_diff <- posterior_sample - evaluation_pt
            estimates <- sapply(Rvec, function(r) {
                a <- sin(r * post_diff)/post_diff
                lpostdens <- log(sum(a)) - (log(n) + log(pi))
                lq - lpostdens
            })
        }
        else {
            x_mat <- matrix(rep(evaluation_pt, n), nrow = n, 
                byrow = TRUE)
            post_diff <- as.matrix(posterior_sample - x_mat)
            estimates <- sapply(Rvec, function(r) {
                a <- abs(rowProds(sin(r * post_diff)/post_diff))
                lpostdens <- log(sum(a)) - (log(n) + ncol(posterior_sample) * 
                  log(pi))
                lq - lpostdens
            })
        }
    }
    else {
        if (!use_kernel_value) {
            posterior_sample.mat <- as.matrix(posterior_sample)
            post_means <- colMeans(posterior_sample.mat)
            post_sds <- colSds(posterior_sample.mat)
            posterior_sample <- as.data.frame(sweep(sweep(posterior_sample.mat, 
                MARGIN = 2, STATS = post_means), MARGIN = 2, 
                STATS = post_sds, FUN = "/"))
            if (!is.numeric(evaluation_pt)) {
                if (!message_suppress) 
                  message("Evaluation point not provided or invalid, using posterior mean")
                evaluation_pt <- if (is.null(dim(posterior_sample))) {
                  0
                }
                else {
                  rep(0, ncol(posterior_sample))
                }
                lq <- posterior_kernel(..., xi = post_means)
            }
            else {
                lq <- posterior_kernel(..., xi = evaluation_pt)
                evaluation_pt <- (evaluation_pt - post_means)/post_sds
            }
            n <- nrow(posterior_sample)
        }
        else {
            evaluation_pt_index <- sample(1:nrow(posterior_sample), 
                size = 1, prob = kernel_value)
            lq <- kernel_value[evaluation_pt_index]
            if (is.null(dim(posterior_sample))) {
                posterior_sample <- posterior_sample[-evaluation_pt_index]
            }
            else {
                posterior_sample <- posterior_sample[-evaluation_pt_index, 
                  ]
            }
        }
        if (is.null(n)) {
            post_diff <- posterior_sample - evaluation_pt
            estimates <- sapply(Rvec, function(r) {
                a <- sin(r * post_diff)/post_diff
                lpostdens <- log(abs(sum(a))) - (log(n) + log(pi))
                lq - lpostdens + log(post_sds)
            })
        }
        else {
            x_mat <- matrix(rep(evaluation_pt, n), nrow = n, 
                byrow = TRUE)
            post_diff <- as.matrix(posterior_sample - x_mat)
            estimates <- sapply(Rvec, function(r) {
                a <- rowProds(sin(r * post_diff)/post_diff)
                lpostdens <- log(abs(sum(a))) - (log(n) + ncol(posterior_sample) * 
                  log(pi))
                lq - lpostdens + sum(log(post_sds))
            })
        }
    }
    estR <- Rvec[which(!is.na(estimates))]
    estValues <- estimates[which(!is.na(estimates))]
    if (Rchangepoint) {
        kcont <- Rchangepoint_continuous
        if (Rchangepoint_continuous) {
            BUGSmodel = "model\n      { # prior\n        beta ~ dnorm(0, 0.001)\n        delta ~ dnorm(0.001,0.001)\n  \n        mu2 ~ dnorm(0, 0.001)\n        tau2 ~ dnorm(0.001,0.001)\n        k ~ dunif(min(R), max(R)) # location of the change point\n  \n        # likelihood\n        for (i in 1:Rlen){\n          J[i] <- step(R[i] - k)\n          mu[i] <- beta * R[i] + J[i] * mu2\n          logtau[i] <- delta * R[i] + J[i] * tau2\n          tau[i] <- exp(logtau[i])\n          chat[i] ~ dnorm(mu[i],tau[i])\n        }\n      }\n      "
            Rlen <- length(estValues)
            data = list(chat = estValues - mean(estValues), Rlen = Rlen, 
                R = estR - mean(estR))
            inits = list(beta = 1, mu2 = 1, tau2 = 1, k = 50)
            parameters = c("beta", "mu2", "delta", "tau2", "k")
            foo <- jags.model(textConnection(BUGSmodel), data = data, 
                inits = inits, n.chains = 1)
            update(foo, burn_in)
            out <- coda.samples(model = foo, variable.names = parameters, 
                n.iter = steps, thin = thin, n.chains = 1)
            if (mean(out[[1]][, "k"]) >= min(estR) && mean(out[[1]][, 
                "k"]) <= max(estR)) {
                index <- which(estR - mean(out[[1]][, "k"]) > 
                  0)[1]
                output <- estValues[index]
                kcont <- TRUE
                if (which(estR - median(out[[1]][, "k"]) > 0)[1] == 
                  1) {
                  warning("The estimates may look like a horizontal band.")
                }
                else if (index/Rlen > 0.75 || Rlen - index < 
                  30) {
                  warning("Too few computed R values. Flat tail check may be unreliable.")
                }
                if (length(Rvec) != Rlen) 
                  warning("some estimates are invalid, posterior predictive estimate unreliable.")
                post_vec <- tabulate(out[[1]][, "k"] + 1, nbins = max(estR))[-(1:min(estR))]
                post_freq <- c(0, rowSums(matrix(post_vec, ncol = Rby, 
                  byrow = TRUE)))
                postpred_estimate = as.numeric((post_freq/sum(post_freq)) \%*\% 
                  estValues)
            }
            else {
                if (!message_suppress) 
                  message("Continuous change-point prior failed, using discrete change-point prior.")
                kcont <- FALSE
            }
        }
        if (!kcont || !Rchangepoint_continuous) {
            kcont <- FALSE
            BUGSmodel = "model\n      { # prior\n        beta ~ dnorm(0, 0.001)\n        delta ~ dnorm(0, 0.001) # as the y-axis is shifted to the change point, (R[i]-R[k]) is negative.\n  \n        mu2 ~ dnorm(0, 0.001)\n        tau2 ~ dnorm(0, 0.001)\n        k ~ dcat(p) # index of the position of the change point\n  \n        # likelihood\n        for (i in 1:Rlen){\n          J[i] <- step(i-k+0.5) #+0.5 due to the case the 'change point' is before the first observation\n          mu[i] <- (1 - J[i]) * (beta * (R[k] - R[i]) + mu2) + J[i] * mu2\n          logtau[i] <- (1 - J[i]) * delta + J[i] * tau2\n          tau[i] <- exp(logtau[i])\n          chat[i] ~ dnorm(mu[i], tau[i])\n        }\n      }\n      "
            Rlen <- length(estValues)
            p <- rep(1/Rlen, Rlen)
            data = list(chat = estValues - mean(estValues), p = p, 
                Rlen = Rlen, R = estR - mean(estR))
            inits = list(beta = 1, mu2 = 1, tau2 = 1, k = 5)
            parameters = c("beta", "mu2", "delta", "tau2", "k")
            foo <- jags.model(textConnection(BUGSmodel), data = data, 
                inits = inits, n.chains = 1)
            update(foo, burn_in)
            out <- coda.samples(model = foo, variable.names = parameters, 
                n.iter = steps, thin = thin, n.chains = 1)
            index <- ifelse(Rone_plus_sd, ceiling(mean(out[[1]][, 
                "k"]) - 1), ceiling(mean(out[[1]][, "k"]) + sd(out[[1]][, 
                "k"]) - 1))
            output <- estValues[index]
            if (ceiling(median(out[[1]][, "k"] - 1)) == 1) {
                warning("The estimates may look like a horizontal band.")
            }
            else if (index/Rlen > 0.75 || Rlen - index < 30) {
                warning("Too few computed R values. Flat tail check may be unreliable.")
            }
            if (index <= 3) 
                warning("Too few estimates before the estimated change point, estimated trend and variability will be poor.")
            post_freq <- tabulate(out[[1]][, "k"], Rlen)
            postpred_estimate = as.numeric((post_freq/sum(post_freq)) \%*\% 
                estValues)
        }
    }
    else {
        distances <- abs(estimates - median(estimates, na.rm = TRUE))
        index <- if (Rmindist) {
            which.min(distances)
        }
        else {
            min(which(distances <= Rtolerance))
        }
        output <- estimates[index]
    }
    if (Rchangepoint) {
        fit <- lm(estValues[index:Rlen] ~ estR[index:Rlen])
    }
    else {
        fit <- lm(estimates[index:length(Rvec)] ~ Rvec[index:length(Rvec)])
    }
    if (plotting) {
        layout(1)
        par(mar = mar)
        plot(Rvec, estimates, pch = 20, pty = "b", main = "FI estimates for each R")
        abline(h = median(estimates[Rvec >= 100], na.rm = TRUE), 
            col = "green4")
        grid()
        if (Rchangepoint) {
            abline(h = postpred_estimate, col = "orange")
            abline(h = output, col = "navy")
            if (kcont) {
                abline(v = mean(out[[1]][, "k"]), col = "blue")
                abline(v = mean(out[[1]][, "k"]) - sd(out[[1]][, 
                  "k"]), col = "blue", lty = "dashed")
                abline(v = mean(out[[1]][, "k"]) + sd(out[[1]][, 
                  "k"]), col = "blue", lty = "dashed")
                abline(h = estValues[which(estR - mean(out[[1]][, 
                  "k"]) + sd(out[[1]][, "k"]) > 0)[1]], col = "navy", 
                  lty = "dashed")
                abline(h = estValues[which(estR - mean(out[[1]][, 
                  "k"]) - sd(out[[1]][, "k"]) > 0)[1]], col = "navy", 
                  lty = "dashed")
            }
            else {
                abline(v = estR[ceiling(mean(out[[1]][, "k"]) - 
                  1)], col = "blue")
                abline(v = estR[ceiling(mean(out[[1]][, "k"]) - 
                  sd(out[[1]][, "k"]) - 1)], col = "blue", lty = "dashed")
                abline(v = estR[ceiling(mean(out[[1]][, "k"]) + 
                  sd(out[[1]][, "k"]) - 1)], col = "blue", lty = "dashed")
                abline(h = estValues[ceiling(mean(out[[1]][, 
                  "k"]) + sd(out[[1]][, "k"]) - 1)], col = "navy", 
                  lty = "dashed")
                abline(h = estValues[ceiling(mean(out[[1]][, 
                  "k"]) - sd(out[[1]][, "k"]) - 1)], col = "navy", 
                  lty = "dashed")
            }
            legend("topright", legend = c("median", "change point R", 
                "change point estimate", "posterior predictive estimate"), 
                col = c("green4", "blue", "navy", "orange"), 
                lwd = 1)
        }
        else {
            legend("topright", legend = "median", col = "green4", 
                lwd = 1)
        }
        mtext(substitute(paste("invalid candidates: ", v, "; total candidates: ", 
            w), list(v = sum(is.na(estimates)), w = length(Rvec))), 
            side = 1, line = 4, col = "red")
        if (Rchangepoint) {
            par(mar = rep(2, 4))
            plot(out)
            autocorr.plot(out)
            invisible(list(Rcandidates = Rvec, result = estimates, 
                posterior_predictive_estimate = postpred_estimate, 
                JAGSsummary = summary(out), JAGSRaftery = raftery.diag(out), 
                #JAGSGeweke = geweke.diag(out)
                ))
        }
        else {
            invisible(list(Rcandidates = Rvec, result = estimates))
        }
    }
    else {
        ifelse(Rchangepoint, list(result = output, selected_R = estR[index]), 
            list(result = output, selected_R = Rvec[index]))
    }
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
