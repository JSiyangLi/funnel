library(data.table)
library(FITSio)
library(dplyr)
library(ggplot2)
library(patchwork)
library(matrixStats)
library(coda)
library(rjags)

##################
### likelihood ###
##################
## constants
c_light <- 2.99792458e8
hubble_constant <- 2.175e-18
#Npos <- 3.6e-41
#Nacc <- 1.44e-48
arm_length <- 2.5e9
fref <- 3e-3
#fref <- 25

#fref = c_light/2/pi/arm_length
#f_star = 25

## parameters:
### Nacc
### Npos
### AECB
### alphaECB
### A1
### alpha1
### A2
### alpha2
### OmegaP
### fp
### rb
### b

## functions
## miscellenous
f_star <- c_light / (2 * pi * arm_length)
W <- function(f) {
  1 - exp(-2 * 1i * f / f_star)
}
fb <- function(rb, fp) rb * fp
### equation 2.6
m <- function(rb, b) (9 * rb^4 + b) / (rb^4 + 1)

## R functions
### equation 2.3
RT <- function(f) {
  (1 / 4032) * (f / f_star)^6 * abs(W(f))^2 * (1 + (5 / 16128) * (f / f_star)^8)^(-1)
}
### equation 2.2
RAE <- function(f) {
  (9 / 20) * abs(W(f))^2 * (1 + (3 * f / (4 * f_star))^2)^(-1)
}

## NPS functions
### equation 2.9
Pa <- function(f, Nacc) {
  (Nacc / (2 * pi * f)^4) * (1 + (4e-4 / f)^2)
}
#Ps <- Npos
### equation 2.8
NXY <- function(f, Nacc, Npos) {
  -(2 * Npos + 8 * Pa(f, Nacc)) * cos(f / f_star) * abs(W(f))^2
}
NX <- function(f, Nacc, Npos) {
  (4 * Npos + 8 * (1 + cos(f / f_star)^2) * Pa(f, Nacc)) * abs(W(f))^2
}
### equation 2.10
NAE <- function(f, Nacc, Npos) NX(f, Nacc, Npos) - NXY(f, Nacc, Npos)
NT <- function(f, Nacc, Npos) NX(f, Nacc, Npos) + 2 * NXY(f, Nacc, Npos)

## Omega prep functions
### equation 2.5
M <- function(f, fp, rb, b) {
  fratio <- f / fp
  final_denominator <- (b + 4 - m(rb, b) + m(rb, b) * fratio^2)
  if(any(is.na(final_denominator)) || any(final_denominator <= 0))
    warning("invalid parameter space: M final denominator")
  fratio^9 * ((1 + rb^4) / (rb^4 + fratio^4))^((9 - b) / 4) * ((b + 4) / final_denominator)^((b + 4) / 2)
}

## Omega functions
### equation 2.4
OmegaPT <- function(f, OmegaP, fp, rb, b) OmegaP * M(f, fp, rb, b)
### equation 2.18
OmegaECB <- function(f, AECB, alphaECB) {
  AECB * (f / fref)^alphaECB
}
### equation 2.15
OmegaDWD <- function(f, A1, A2, alpha1, alpha2) {
  (A1 * (f / f_star)^alpha1) / (1 + A2 * (f / f_star)^alpha2)
}
### equation 2.19
Omegagw <- function(f, A1, A2, AECB, alpha1, alpha2, alphaECB, OmegaP, fp, rb, b) {
  if (any(is.na(OmegaPT(f, OmegaP, fp, rb, b))))
    warning("OmegaPT is NA")
  OmegaPT(f, OmegaP, fp, rb, b) + OmegaECB(f, AECB, alphaECB) + OmegaDWD(f, A1, A2, alpha1, alpha2)
}

## likelihood and prior on the original scale
### equation 2.21
C <- function(f, Nacc, Npos, AECB, alphaECB, A1, alpha1, A2, alpha2, OmegaP, fp, rb, b) {
  # gives the diagonal values of matrix C in the paper
  AE_channel <- NAE(f, Nacc, Npos) + 3 * hubble_constant^2 / (4 * pi^2 * f^3) * RAE(f) * Omegagw(f, A1, A2, AECB, alpha1, alpha2, alphaECB, OmegaP, fp, rb, b)
  T_channel <- NT(f, Nacc, Npos)
  cbind(AE_channel, AE_channel, T_channel)
}
transit_loglikelihood <- function(periodogram, f, Nacc, Npos, AECB, alphaECB, A1, alpha1, A2, alpha2, OmegaP, fp, rb, b) {
  covelements <- C(f, Nacc, Npos, AECB, alphaECB, A1, alpha1, A2, alpha2, OmegaP, fp, rb, b)
  N <- nrow(periodogram)
  ker <- -rowSums(periodogram / (2 * covelements)) # summing over all 3 channels
  normalisationconstant <- -0.5 * sum(log(covelements)) - N * 3 * log(pi) - N * 1.5 * log(2) # the determinant of a diagonal matrix is the product of diagonal
  # the sum() here sums over all the data and all 3 channels
  if (is.na(normalisationconstant))
    warning("normalisation constant is NA")
  if (any(is.na(ker)))
    warning("kernel is NA")
  normalisationconstant + sum(ker) # summing over all data
}
#! the control model simply ignores OmegaPT function in Omegagw function
Omegagw_control <- function(f, A1, A2, AECB, alpha1, alpha2, alphaECB) {
  OmegaECB(f, AECB, alphaECB) + OmegaDWD(f, A1, A2, alpha1, alpha2)
}
C_control <- function(f, Nacc, Npos, AECB, alphaECB, A1, alpha1, A2, alpha2) {
  AE_channel <- NAE(f, Nacc, Npos) + 3 * hubble_constant^2 / (4 * pi^2 * f^3) * RAE(f) * Omegagw_control(f, A1, A2, AECB, alpha1, alpha2, alphaECB)
  T_channel <- NT(f, Nacc, Npos)
  cbind(AE_channel, AE_channel, T_channel)
}
control_loglikelihood <- function(periodogram, f, Nacc, Npos, AECB, alphaECB, A1, alpha1, A2, alpha2) {
  covelements <- C_control(f, Nacc, Npos, AECB, alphaECB, A1, alpha1, A2, alpha2)
  N <- nrow(periodogram)
  ker <- -rowSums(periodogram / (2 * covelements)) # summing over all 3 channels
  normalisationconstant <- -0.5 * (sum(log(covelements))) - N * 3 * log(pi) - N * 1.5 * log(2) # the determinant of a diagonal matrix is the product of diagonal
  normalisationconstant + sum(ker) # summing over all data
}


## posterior kernels
## a more computationally stable implementation
transit_logprior_log <- function(logNacc, logNpos, logAECB, alphaECB, logA1, alpha1, logA2, alpha2, logOmegaP, logfp, rb, b) {
  # normal prior on log scale of all N, A, Omega and f parameters, centered at true value
  theta1caron <- c(logNacc, logNpos, logAECB, logA1, logA2, logOmegaP, logfp)
  true_theta1caron <- log(c(1.44e-48, 3.6e-41, 5.6e-12, 7.44e-14, 2.96e-7, 1e-9, 0.002))
  theta1_dens <- sum(mapply(dnorm, x = theta1caron, mean = true_theta1caron, log = TRUE))
  
  # normal prior on ordinary scale of alpha, r and b parameters, centered at true value
  theta2 <- c(alphaECB, alpha1, alpha2, rb, b)
  true_theta2 <- c(2/3, -1.98, -2.6, 0.4, 1)
  theta2_dens <- sum(mapply(dnorm, x = theta2, mean = true_theta2, log = TRUE))
  
  theta1_dens + theta2_dens
}
transit_q_log <- function(periodogram, f, xicaron) { # corresponds to pi_caron density
  logNacc = xicaron[1]
  logNpos = xicaron[2]
  logAECB = xicaron[3]
  alphaECB = xicaron[4]
  logA1 = xicaron[5]
  alpha1 = xicaron[6]
  logA2 = xicaron[7]
  alpha2 = xicaron[8]
  logOmegaP = xicaron[9]
  logfp = xicaron[10]
  rb = xicaron[11]
  b = xicaron[12]
  
  transit_loglikelihood(periodogram, f, exp(logNacc), exp(logNpos), exp(logAECB), alphaECB, exp(logA1), alpha1, exp(logA2), alpha2, exp(logOmegaP), exp(logfp), rb, b) + 
    transit_logprior_log(logNacc, logNpos, logAECB, alphaECB, logA1, alpha1, logA2, alpha2, logOmegaP, logfp, rb, b)
}
control_logprior_log <- function(logNacc, logNpos, logAECB, alphaECB, logA1, alpha1, logA2, alpha2) {
  # normal prior on log scale of all N, A, Omega and f parameters, centered at true value
  theta1caron <- c(logNacc, logNpos, logAECB, logA1, logA2)
  true_theta1caron <- log(c(1.44e-48, 3.6e-41, 5.6e-12, 7.44e-14, 2.96e-7))
  theta1_dens <- sum(mapply(dnorm, x = theta1caron, mean = true_theta1caron, log = TRUE))
  
  # normal prior on ordinary scale of alpha, r and b parameters, centered at true value
  theta2 <- c(alphaECB, alpha1, alpha2)
  true_theta2 <- c(2/3, -1.98, -2.6)
  theta2_dens <- sum(mapply(dnorm, x = theta2, mean = true_theta2, log = TRUE))
  
  theta1_dens + theta2_dens
}
control_q_log <- function(periodogram, f, xicaron) { # corresponds to pi_caron density
  logNacc = xicaron[1]
  logNpos = xicaron[2]
  logAECB = xicaron[3]
  alphaECB = xicaron[4]
  logA1 = xicaron[5]
  alpha1 = xicaron[6]
  logA2 = xicaron[7]
  alpha2 = xicaron[8]
  
  control_loglikelihood(periodogram, f, exp(logNacc), exp(logNpos), exp(logAECB), alphaECB, exp(logA1), alpha1, exp(logA2), alpha2) + 
    control_logprior_log(logNacc, logNpos, logAECB, alphaECB, logA1, alpha1, logA2, alpha2)
}

###############################
### case specific functions ###
###############################
# cases:
fp_candidates <- c(1e-05, 1e-04, 0.001, 0.01, 0.1, 1, 0.002, 3e-04, 0.003, 6e-04, 0.006)
OmegaP_candidates <- c(1e-09, 1e-10)
#1. data and posterior sample preparation
###############################
LISAprepare <- function(OmegaP, fp) {
  #################
  ### read data ###
  #################
  y <- t(fread(paste0(getwd(), "\\PhaseTransition_fs\\", OmegaP, "\\", fp, "\\data.txt")))
  transit_post.fits <- readFITS(here::here(paste0("PhaseTransition_fs\\", OmegaP, "\\", fp, "\\Chains")))
  control_post.fits <- readFITS(here::here(paste0("PhaseTransition_fs\\", OmegaP, "\\", fp, "\\Control\\Chains")))
  transit_post_points.df <- transit_post.fits$imDat |> as.data.frame()
  colnames(transit_post_points.df) <- read.table(paste0(getwd(), "\\PhaseTransition_fs\\TransitionName.txt")) |> t()
  transit_post.df <- transit_post_points.df[, c(1, 2, 7, 8, 3, 4, 5, 6, 9, 10, 11, 12)] # adjusting the order of inputs to fit with the rest of codes
  
  control_post_points.df <- control_post.fits$imDat |> as.data.frame()
  colnames(control_post_points.df) <- read.table(paste0(getwd(), "\\PhaseTransition_fs\\ControlName.txt")) |> t()
  control_post.df <- control_post_points.df[, c(1, 2, 7, 8, 3, 4, 5, 6)] # adjusting the order of inputs to fit with the rest of codes
  
  ######################################
  ### preparing data and true values ###
  ######################################
  # preparing transformed posterior sample and true value vector
  # Nacc, Npos, AECB, alphaECB, A1, alpha1, A2, alpha2, OmegaP, fp, rb, b

  # true values
  x <- c(log(1.44e-48), log(3.6e-41), log(5.6e-12), 2/3, log(7.44e-14), -1.98, log(2.96e-7), -2.6, log(OmegaP), log(fp), 0.4, 1)
  # transition model burn-in removal
  transit_transformed.post.df <- cbind(log(transit_post.df[-(1:999), 1:3]), transit_post.df[-(1:999), 4], log(transit_post.df[-(1:999), 5]), transit_post.df[-(1:999), 6], 
                                       log(transit_post.df[-(1:999), 7]), transit_post.df[-(1:999), 8], log(transit_post.df[-(1:999), 9:10]), transit_post.df[-(1:999), 11:12])
  # control model burn-in removal
  control_transformed.post.df <- cbind(log(control_post.df[-(1:999), 1:3]), control_post.df[-(1:999), 4], 
                                       log(control_post.df[-(1:999), 5]), control_post.df[-(1:999), 6], log(control_post.df[-(1:999), 7]), control_post.df[-(1:999), 8])
  
  colnames(transit_transformed.post.df)[4:8] <- c("alpha3", "A1", "alpha1", "A2", "alpha2")
  colnames(control_transformed.post.df)[4:8] <- c("alpha3", "A1", "alpha1", "A2", "alpha2")
  # function output
  list(y = y, # simulated periodogram
       x = x, # underlying true parameter values
       transit_transformed.post.df = transit_transformed.post.df, # transition model posterior sample
       control_transformed.post.df = control_transformed.post.df  # control model posterior sample
  ) 
}

####################
#2. making plots for each case
####################
LISAplot <- function(OmegaP, fp, ggplot = TRUE, combined = TRUE, gglwd = 0.25) {
  a <- LISAprepare(OmegaP, fp, standardise = FALSE)
  y <- a$y
  x <- a$x
  
  ###########################
  ### plots of model fits ###
  ###########################
  if (!ggplot) { # base R plot
    plot(log10(y[, 1]), log10(y[, 4]), type = "l", yaxt = "n",
         main = "T channel", ylab = "log10periodogram", xlab = "log10frequency")
    axis(2, at = -55:-39, las = 1)
    lines(log10(y[, 1]), log10(NT(y[, 1], exp(x[1]), exp(x[2]))), col = "red")
    plot(log10(y[, 1]), log10(y[, 2]), type = "l", yaxt = "n",
         main = "A channel", ylab = "log10periodogram", xlab = "log10frequency")
    axis(2, at = -48:-33, las = 1)
    lines(log10(y[, 1]), log10(C(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                 x[4], exp(x[5]), x[6], exp(x[7]), x[8], exp(x[9]),
                                 exp(x[10]), 
                                 x[11], x[12])[, 1]), col = "red")
    lines(log10(y[, 1]), log10(C_control(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                         x[4], exp(x[5]), x[6], exp(x[7]), x[8])[, 1]), col = "blue")
    abline(v = log10(0.005), col = "green")
    
    plot(log10(y[, 1]), log10(Omegagw(y[, 1], AECB = exp(x[3]),
                                      alphaECB = x[4], A1 = exp(x[5]), alpha1 = x[6],
                                      A2 = exp(x[7]), alpha2 = x[8], OmegaP = exp(x[9]),
                                      fp = exp(x[10]), rb = x[11], b = x[12])), type = "l",
         ylab = "log10 Omegagw", xlab = "log10 frequency", main = "transition Omegagw", las = 1)
    abline(v = log10(0.002), col = "green")
  } else { # ggplot
    # A channel
    Aperiodogram <- cbind(log10(y[, 1]), log10(y[, 2])) %>% as.data.frame()
    colnames(Aperiodogram) <- c("log10 frequency", "log10 periodogram")
    Atransit_fitted <- cbind(log10(y[, 1]), log10(C(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                                    x[4], exp(x[5]), x[6], exp(x[7]), x[8], exp(x[9]),
                                                    exp(x[10]), x[11], x[12])[, 1])) %>% as.data.frame()
    colnames(Atransit_fitted) <- c("log10 frequency", "log10 periodogram")
    Acontrol_fitted <- cbind(log10(y[, 1]), log10(C_control(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                                            x[4], exp(x[5]), x[6], exp(x[7]), x[8])[, 1])) %>% as.data.frame()
    colnames(Acontrol_fitted) <- c("log10 frequency", "log10 periodogram")
    Aperiodogram.gg <- Aperiodogram %>%
      ggplot(mapping = aes(x = `log10 frequency`, y = `log10 periodogram`)) +
      geom_line() +
      geom_line(data = Acontrol_fitted, color = "blue", linewidth = gglwd) +
      geom_line(data = Atransit_fitted, color = "red", linewidth = gglwd) +
      ggtitle("A channel")
    # E channel
    Eperiodogram <- cbind(log10(y[, 1]), log10(y[, 3])) %>% as.data.frame()
    colnames(Eperiodogram) <- c("log10 frequency", "log10 periodogram")
    Etransit_fitted <- cbind(log10(y[, 1]), log10(C(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                                    x[4], exp(x[5]), x[6], exp(x[7]), x[8], exp(x[9]),
                                                    exp(x[10]), x[11], x[12])[, 2])) %>% as.data.frame()
    colnames(Etransit_fitted) <- c("log10 frequency", "log10 periodogram")
    Econtrol_fitted <- cbind(log10(y[, 1]), log10(C_control(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                                            x[4], exp(x[5]), x[6], exp(x[7]), x[8])[, 2])) %>% as.data.frame()
    colnames(Econtrol_fitted) <- c("log10 frequency", "log10 periodogram")
    Eperiodogram.gg <- Eperiodogram %>%
      ggplot(mapping = aes(x = `log10 frequency`, y = `log10 periodogram`)) +
      geom_line() +
      geom_line(data = Econtrol_fitted, color = "blue", linewidth = gglwd) +
      geom_line(data = Etransit_fitted, color = "red", linewidth = gglwd) +
      ggtitle("E channel")
    
    if (combined) { # all 3 ggplots combines into one
      # T channel
      Tperiodogram <- cbind(log10(y[, 1]), log10(y[, 4]),
                            log10(C_control(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                            x[4], exp(x[5]), x[6], exp(x[7]), x[8])[, 3]),
                            log10(C(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                    x[4], exp(x[5]), x[6], exp(x[7]), x[8], exp(x[9]),
                                    exp(x[10]), x[11], x[12])[, 3])) %>% as.data.frame()
      colnames(Tperiodogram) <- c("log10 frequency", "log10 periodogram", "control model", "transition model")
      Tperiodogram.gg <- Tperiodogram %>%
        as.data.table() %>%
        melt(id = 1) %>%
        ggplot(mapping = aes(x = `log10 frequency`, y = value, color = variable)) +
        geom_line(linewidth = gglwd) +
        ggtitle("T channel") +
        ylab("log10 periodogram") +
        scale_color_manual(values = c("black", "blue", "red")) +
        theme(legend.position = "bottom",
              legend.title = element_blank())
      
      (Aperiodogram.gg / Eperiodogram.gg / Tperiodogram.gg) +
        plot_annotation(title = substitute(expression(paste(Omega[p], "=", v, ", ", f[p], "=", w)), list(v = OmegaP, w = fp)))
    } else { # one ggplot for each channel
      # T channel
      Tperiodogram <- cbind(log10(y[, 1]), log10(y[, 4])) %>% as.data.frame()
      colnames(Tperiodogram) <- c("log10 frequency", "log10 periodogram")
      Ttransit_fitted <- cbind(log10(y[, 1]), log10(C(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                                      x[4], exp(x[5]), x[6], exp(x[7]), x[8], exp(x[9]),
                                                      exp(x[10]), x[11], x[12])[, 3])) %>% as.data.frame()
      colnames(Ttransit_fitted) <- c("log10 frequency", "log10 periodogram")
      Tcontrol_fitted <- cbind(log10(y[, 1]), log10(C_control(y[, 1], exp(x[1]), exp(x[2]), exp(x[3]),
                                                              x[4], exp(x[5]), x[6], exp(x[7]), x[8])[, 3])) %>% as.data.frame()
      colnames(Tcontrol_fitted) <- c("log10 frequency", "log10 periodogram")
      Tperiodogram.gg <- Tperiodogram %>%
        ggplot(mapping = aes(x = `log10 frequency`, y = `log10 periodogram`)) +
        geom_line() +
        geom_line(data = Tcontrol_fitted, color = "blue", linewidth = gglwd) +
        geom_line(data = Ttransit_fitted, color = "red", linewidth = gglwd) +
        ggtitle("T channel")
      
      Aperiodogram.gg
      Eperiodogram.gg
      Tperiodogram.gg
    }
  }
}


#mapply(LISAplot, fp_candidates, OmegaP_candidates)
#sapply(LISAplot, fp_candidates[1:4], OmegaP = 1e-11)
# sapply decided not to work
pdf("AETeach.pdf", width = 5, height = 10)
#1e-09
LISAplot(1e-09, 1e-04)
LISAplot(1e-09, 0.001)
LISAplot(1e-09, 0.01)
LISAplot(1e-09, 0.1)
LISAplot(1e-09, 0.002)
LISAplot(1e-09, 3e-04)
LISAplot(1e-09, 0.003)
LISAplot(1e-09, 6e-04)
LISAplot(1e-09, 0.006)
LISAplot(1e-09, 1)
LISAplot(1e-09, 1e-05)

#1e-10
LISAplot(1e-10, 1e-04)
LISAplot(1e-10, 0.001)
LISAplot(1e-10, 0.01)
LISAplot(1e-10, 0.1)
LISAplot(1e-10, 0.002)
LISAplot(1e-10, 3e-04)
LISAplot(1e-10, 0.003)
LISAplot(1e-10, 6e-04)
LISAplot(1e-10, 0.006)
LISAplot(1e-10, 1)
LISAplot(1e-10, 1e-05)

#1e-11
LISAplot(1e-11, 1e-04)
LISAplot(1e-11, 0.001)
LISAplot(1e-11, 0.01)
LISAplot(1e-11, 0.1)
LISAplot(1e-11, 1)
LISAplot(1e-11, 1e-05)
dev.off()


##########################################
## Fourier Integral function (general) ###
##########################################
FI <- function(evaluation_pt = NULL, # point of evaluation
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

FIautoR <- function(# basic functionality arguments
                    evaluation_pt = NULL,
                    use_kernel_value = FALSE, # use kernel value or kernel function for the computation?
                    posterior_kernel,
                    posterior_sample,
                    kernel_value, # enabling the function to compute FI without knowing the posterior kernel function
                    # standardising the posterior sample?
                    standardisation = TRUE,
                    # R candidates creation
                    Rmin = 10, # minimum R value
                    Rby = 10, # step size in the R vector
                    Rmax = 2500, # maximum value of the R vector
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
  if (!is.data.frame(posterior_sample))
    stop("The posterior sample needs to be in a data frame")
  if (!is.numeric(Rby) || !is.numeric(Rmax))
    stop("The limiting factor quantities needs to be numeric")
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
      
      Rvec <- seq(Rmin, Rmax, Rby)
      n <- nrow(posterior_sample)
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
    if (is.null(n)) { # the parameter space is univariate
      post_diff <- posterior_sample - evaluation_pt
      estimates <- sapply(Rvec, function(r) {
        a <- sin(r * post_diff) / post_diff
        lpostdens <- log(sum(a)) - (log(n) + log(pi)) # Fourier integral theorem
        lq - lpostdens
      })
    } else { # higher-dim parameter space
      x_mat <- matrix(rep(evaluation_pt, n), nrow = n, byrow = TRUE)
      post_diff <- as.matrix(posterior_sample - x_mat)
      estimates <- sapply(Rvec, function(r) {
        a <- abs(rowProds(sin(r * post_diff) / post_diff))
        lpostdens <- log(sum(a)) - (log(n) + ncol(posterior_sample) * log(pi)) # Fourier integral theorem
        lq - lpostdens
      })
    }
  } else { # standardising the posterior sample
    # evaluating posterior density
    if (!use_kernel_value) { # use the posterior kernel function
      # standardization
      posterior_sample.mat <- as.matrix(posterior_sample)
      post_means <- colMeans(posterior_sample.mat)
      post_sds <- colSds(posterior_sample.mat)
      posterior_sample <- sweep(posterior_sample.mat, MARGIN = 2, STATS = post_means) |>
        sweep(MARGIN = 2, STATS = post_sds, FUN = "/") |> 
        as.data.frame()
      # setting default evaluation point to the posterior mean
      if (!is.numeric(evaluation_pt)) {
        if (!message_suppress) message("Evaluation point not provided or invalid, using posterior mean")
        
        evaluation_pt <- if (is.null(dim(posterior_sample))) {0} else {rep(0, ncol(posterior_sample))}
        lq <- posterior_kernel(..., xi = post_means) # evaluating at the posterior mean
      } else { # transformed evaluation point is specified
        lq <- posterior_kernel(..., xi = evaluation_pt)
        evaluation_pt <- (evaluation_pt - post_means) / post_sds
      }
      
      Rvec <- seq(Rmin, Rmax, Rby)
      n <- nrow(posterior_sample)
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
    if (is.null(n)) { # the parameter space is univariate
      post_diff <- posterior_sample - evaluation_pt
      estimates <- sapply(Rvec, function(r) {
        a <- sin(r * post_diff) / post_diff
        lpostdens <- log(abs(sum(a))) - (log(n) + log(pi)) # Fourier integral theorem
        lq - lpostdens + log(post_sds)
      })
    } else { # higher-dim parameter space
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
        post_freq <- c(0, rowSums(matrix(post_vec, ncol = Rby, byrow = TRUE)))
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
      invisible(list("Rcandidates" = Rvec, "result" = estimates,
                     "posterior_predictive_estimate" = postpred_estimate,
                     "JAGSsummary" = summary(out), "JAGSRaftery" = raftery.diag(out),
                     "JAGSGeweke" = geweke.diag(out)))
    } else { # median output
      invisible(list("Rcandidates" = Rvec, "result" = estimates))
    }
  } else { # no detailed results
    ifelse(Rchangepoint, list("result" = output, "selected_R" = estR[index]), list("result" = output, "selected_R" = Rvec[index]))
  }
}
FIautoR(posterior_kernel = transit_q_log, posterior_sample = transit_transformed.post.df, 
        periodogram = y[, 2:4], f = y[, 1], plotting = TRUE)
###############
#3. LISA FI
################
LISAFI <- function(OmegaP, fp, message_suppress = TRUE, plotting = FALSE,
                   Rmin = 5, Rby = 5, Rmax = 500, burn_in = 1000, steps = 5000, thin = 1, 
                   Rchangepoint = TRUE, Rchangepoint_continuous = FALSE,
                   evaluation_pt = NULL, standardisation = TRUE) {
  b <- LISAprepare(OmegaP, fp)
  y <- b$y
  transit_transformed.post.df <- b$transit_transformed.post.df
  control_transformed.post.df <- b$control_transformed.post.df
  
  print(paste0("transition model: OmegaP = ", OmegaP, ", fp = ", fp))
  set.seed(42)
  transit_log_c <- FIautoR(posterior_kernel = transit_q_log, posterior_sample = transit_transformed.post.df, standardisation = standardisation,
                           periodogram = y[, 2:4], f = y[, 1], evaluation_pt = evaluation_pt,
                           message_suppress = message_suppress, Rchangepoint_continuous = Rchangepoint_continuous, Rchangepoint = Rchangepoint,
                           Rmin = Rmin, Rby = Rby, Rmax = Rmax, burn_in = burn_in, steps = steps, thin = thin,
                           plotting = plotting)
  print(paste0("control model: OmegaP = ", OmegaP, ", fp = ", fp))
  set.seed(42)
  control_log_c <- FIautoR(posterior_kernel = control_q_log, posterior_sample = control_transformed.post.df, standardisation = standardisation,
                           periodogram = y[, 2:4], f = y[, 1], evaluation_pt = evaluation_pt[1:8],
                           message_suppress = message_suppress, Rchangepoint_continuous = Rchangepoint_continuous, Rchangepoint = Rchangepoint,
                           Rmin = Rmin, Rby = Rby, Rmax = Rmax, burn_in = burn_in, steps = steps, thin = thin,
                           plotting = plotting)
  c(transit = transit_log_c, control = control_log_c)
}

LISAsingleR <- function(OmegaP, fp, evaluation_pt = NULL, transitR = 50, controlR = 50, message_suppress = FALSE) {
  b <- LISAprepare(OmegaP, fp)
  y <- b$y
  transit_transformed.post.df <- b$transit_transformed.post.df
  control_transformed.post.df <- b$control_transformed.post.df
  
  transit_log_c <- FI(posterior_kernel = transit_q_log, posterior_sample = transit_transformed.post.df, 
                      R = transitR, periodogram = y[, 2:4], f = y[, 1], evaluation_pt = evaluation_pt,
                      message_suppress = message_suppress)
  control_log_c <- FI(posterior_kernel = control_q_log, posterior_sample = control_transformed.post.df,
                      R = controlR, periodogram = y[, 2:4], f = y[, 1], evaluation_pt = evaluation_pt,
                      message_suppress = message_suppress)
  list(transit = transit_log_c, control = control_log_c)
}

###########################################
### actual FI estimation and processing ###
###########################################
# FI
ptm1 <- proc.time()
fi09 <- sapply(fp_candidates, LISAFI, OmegaP = 1e-09)
fi10 <- sapply(fp_candidates, LISAFI, OmegaP = 1e-10)

# specific case: posterior mean gives invalid M final denominator: evaluating at a different point
b <- LISAprepare(1e-11, 1e-04)
y <- b$y
transit_transformed.post.df <- b$transit_transformed.post.df
x_transit <- colMeans(transit_transformed.post.df)
fi112 <- LISAFI(1e-11, 1e-04, evaluation_pt = x_transit + c(rep(0, 9), -1, 0, 0))

b <- LISAprepare(1e-11, 1e-05)
y <- b$y
transit_transformed.post.df <- b$transit_transformed.post.df
x_transit <- colMeans(transit_transformed.post.df)
fi111 <- LISAFI(1e-11, 1e-05, Rmax = 200, evaluation_pt = x_transit + c(rep(0, 9), -5, rep(0, 2)))

fi113 <- sapply(fp_candidates[3:6], LISAFI, OmegaP = 1e-11)
fi11 <- cbind(fi111, fi112, fi113)

ptm_fi <- proc.time() - ptm1

# a different result as using DIC
L = LISAFI(1e-09, 1e-04, plotting = TRUE, Rmin = 5, Rmax = 500, Rby = 5) # reasonable except variances seem to grow quite quickly.


# log marginal likelihood differences
fi09mat <- matrix(unlist(fi09), nrow = 2)
fi10mat <- matrix(unlist(fi10), nrow = 2)
fi11mat <- matrix(unlist(fi11), nrow = 2)
chat_diff09 <- -colDiffs(fi09mat) * log10(exp(1))
chat_diff10 <- -colDiffs(fi10mat) * log10(exp(1))
chat_diff11 <- -colDiffs(fi11mat) * log10(exp(1))

# plotting
pdf("evidencefig.pdf", height = 5, width = 8)
par(pch = 19, las = 1)
plot(log10(fp_candidates), chat_diff09, type = "n",
     xlab = expression(paste(log[10], "(", f[p], ")")), 
     ylab = expression(paste("Difference in ", log[10], "(evidence)")))
grid()
rect(-5.2, 400, -4.2, 600, border = NA, col = "grey90")
rect(-5.2, -15, 0.2, 15, border = NA, col = "grey70")
#abline(h = 0, col = "red")
points(log10(fp_candidates), chat_diff09, col = "red")
points(log10(fp_candidates), chat_diff10, col = "blue")
points(log10(fp_candidates[1:6]), chat_diff11, col = "green4")
legend("topleft", col = c("red", "blue", "green4"),
       legend = c(substitute(paste(Omega[p], " = ", v), list(v = 10^-9)),
                  substitute(paste(Omega[p], " = ", w), list(w = 10^-10)),
                  substitute(paste(Omega[p], " = ", x), list(x = 10^-11))),
       pch = 19, bty = "n")

plot(log10(fp_candidates), chat_diff09, col = "red",
     ylim = c(-15, 15), yaxt = "n",
     xlab = expression(paste(log[10], "(", f[p], ")")), 
     ylab = expression(paste("Difference in ", log[10], "(evidence)")))
points(log10(fp_candidates), chat_diff10, col = "blue")
points(log10(fp_candidates[1:6]), chat_diff11, col = "green4")
axis(2, at = c(-15, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 15))
grid()
abline(h = -2)
abline(h = 2)
abline(h = -1, lty = "dotdash")
abline(h = 1, lty = "dotdash")
abline(h = -0.5, lty = "dotted")
abline(h = 0.5, lty = "dotted")
abline(h = 0, col = "red")

diff11 <- chat_diff11[order(fp_candidates[1:6])]
deltaevi <- t(rbind(sort(fp_candidates), 
                    chat_diff09[order(fp_candidates)], 
                    chat_diff10[order(fp_candidates)],
                    NA))
deltaevi[c(1, 2, 5, 9, 10, 11), 4] <- diff11
colnames(deltaevi) <- c("log10fp", "OmegaP=10^-9", "OmegaP=10^-10", "OmegaP=10^-11")
deltaevi
dev.off()

############################
### change point problem ###
############################ if the number of valid estimates >= 3 to estimate the mean component, or >=5 for the whole model
L = LISAFI(1e-09, 6e-04, plotting = TRUE)
# discrete change-point prior
LtransitR <- L$transit.Rcandidates[which(!is.na(L$transit.result))]
Ltransit <- L$transit.result[which(!is.na(L$transit.result))]
LcontrolR <- L$control.Rcandidates[which(!is.na(L$control.result))]
Lcontrol <- L$control.result[which(!is.na(L$control.result))]

BUGSmodel = "model
{ # prior
  beta ~ dnorm(0, 0.001)
  delta ~ dgamma(0.001,0.001) # as the y-axis is shifted to the change point, (R[i]-R[k]) is negative.
  #delta <- -negdelta
  sigma1 <- 1 / delta
  
  mu2 ~ dnorm(0, 0.001)
  tau2 ~ dgamma(0.001,0.001)
  sigma2 <- 1 / tau2
  k ~ dcat(p) # index of the position of the change point
  
  # likelihood
  for (i in 1:Rlen){
    J[i] <- step(i-k+0.5) #+0.5 due to the case the 'change point' is before the first observation
    mu[i] <- (1 - J[i]) * (beta * (R[k] - R[i]) + mu2) + J[i] * mu2
    tau[i] <- (1 - J[i]) * (delta * (R[k] - R[i]) + tau2) + J[i] * tau2
    chat[i] ~ dnorm(mu[i], tau[i])
  }
}
"
#log(sigma) = alpha + beta R, alpha, beta ~ N
desc_time <- proc.time()
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
    logtau[i] <- (1 - J[i]) * delta + I[i] * tau2
    tau[i] <- exp(logtau[i])
    chat[i] ~ dnorm(mu[i], tau[i])
  }
}
"
Rlen <- length(Ltransit)
p <- rep(1/Rlen, Rlen)
# The data (use NA for no data) as a list
data = list(chat = Ltransit - mean(Ltransit), p = p, Rlen = Rlen, R = LtransitR - mean(LtransitR))
# The initial values as a list (JAGS will automatically generate these if unspecified)
inits = list(beta = 1, mu2 = 1, tau2 = 1, k = 5)
# parameters to monitor
parameters = c('beta', 'mu2', 'delta', 'tau2', 'k')
# How many burn-in steps?
burn_in = 1000
# How many proper steps?
steps = 10000
# Thinning?
thin = 2
# Random number seed
seed = 42
#compilation of the BUGS model, no Gibbs sampling yet
foo <- jags.model(textConnection(BUGSmodel),data=data, inits=inits, n.chains=1)
#burnin samples
update(foo, burn_in)
#draws n.iter MCMC samples, monitors the parameters specified in variable.names, thins the
#output by a factor of thin and stores everything in a MCMC.list object
out <- coda.samples(model=foo, variable.names=parameters, n.iter=steps, thin=thin, n.chains=1)

sdesc_time <- proc.time() - desc_time

#Obtain summary statistics for each of the monitored parameters
summary(out)
#Obtain traceplots and kernel density estimates for each parameter
par(mar = rep(2, 4)) #adjusts margin plotting parameters
plot(out)
autocorr.plot(out)
raftery.diag(out)
geweke.diag(out)

# flat tail check
fit <- lm(Ltransit[ceiling(mean(out[[1]][, "k"] - 1)):length(Ltransit)] ~ LcontrolR[ceiling(mean(out[[1]][, "k"] - 1)):length(Ltransit)])
plot(fit, which = 1)
summary(fit)
(Ltransit[ceiling(mean(out[[1]][, "k"] - 1))] - mean(Ltransit) - mean(out[[1]][, "k"])) / mean(sqrt(out[[1]][, 5]))
Ltransit[ceiling(mean(out[[1]][, "k"] - 1))] - median(Ltransit)

# posterior mean of evaluated point
(tabulate(out[[1]][, "k"], nbins = Rlen) / length(out[[1]][, "k"])) %*% Ltransit

#variance parameter plot
dev.off()
plot.new()
plot.window(xlim = c(0, max(out[[1]][, "k"])), ylim = c(0, 1 / min(out[[1]][, 5])))
axis(1)
axis(2)
sapply(1:5000, function(i) 
  lines(c(0, out[[1]][i, 2]), c(1 / out[[1]][i, 5] - out[[1]][i, 2] * 1 / out[[1]][i, 4], 1 / out[[1]][i, 5]), col = '#FF000088'))
title(xlab = "index of R", ylab = "precision")

plot.new()
plot.window(xlim = c(0, max(out[[1]][, "k"])), ylim = c(0, max(out[[1]][, 5])))
axis(1)
axis(2)
sapply(1:5000, function(i) 
  lines(c(0, out[[1]][i, 2]), c(1 / (1 / out[[1]][i, 5] - out[[1]][i, 2] * 1 / out[[1]][i, 4]), out[[1]][i, 5]), col = '#FF000088'))
title(xlab = "index of R", ylab = "variance")

LtransitR[13]
LcontrolR[14]
## continuous change-point prior
cont_time <- proc.time()
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
Rlen <- length(Ltransit)
# The data (use NA for no data) as a list
data = list(chat = Ltransit - mean(Ltransit), Rlen = Rlen, R = LtransitR - mean(LtransitR))
# The initial values as a list (JAGS will automatically generate these if unspecified)
inits = list(beta = 1, mu2 = 1, tau2 = 1, k = 50)
# parameters to monitor
parameters = c('beta', 'mu2', 'delta', 'tau2', 'k')
# How many burn-in steps?
burn_in = 1000
# How many proper steps?
steps = 5000
# Thinning?
thin = 1
# Random number seed
seed = 42
#compilation of the BUGS model, no Gibbs sampling yet
foo <- jags.model(textConnection(BUGSmodel),data=data, inits=inits, n.chains=1)
#burnin samples
update(foo, burn_in)
#draws n.iter MCMC samples, monitors the parameters specified in variable.names, thins the
#output by a factor of thin and stores everything in a MCMC.list object
out <- coda.samples(model=foo, variable.names=parameters, n.iter=steps, thin=thin, n.chains=1)

scont_time <- proc.time() - cont_time

#Obtain summary statistics for each of the monitored parameters
summary(out)
#Obtain traceplots and kernel density estimates for each parameter
par(mar = rep(2, 4)) #adjusts margin plotting parameters
plot(out)
autocorr.plot(out)
raftery.diag(out)
geweke.diag(out)

# flat tail check
index <- which(LtransitR - mean(out[[1]][, "k"]) > 0)[1]
fit <- lm(Ltransit[index:length(Ltransit)] ~ LtransitR[index:length(Ltransit)])
plot(fit, which = 1)
summary(fit)


post_vec <- tabulate(out[[1]][, "k"] + 1, nbins = max(LtransitR))[-(1:min(LtransitR))]
post_freq <- c(0, rowSums(matrix(post_vec, ncol = 5, byrow = TRUE)))
as.numeric((post_freq / sum(post_freq)) %*% Ltransit)

###########
### DIC ###
###########












