\name{FI}
\alias{FI}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
FI(evaluation_pt = NULL, posterior_kernel, posterior_sample, R = 50, message_suppress = FALSE, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{evaluation_pt}{
%%     ~~Describe \code{evaluation_pt} here~~
}
  \item{posterior_kernel}{
%%     ~~Describe \code{posterior_kernel} here~~
}
  \item{posterior_sample}{
%%     ~~Describe \code{posterior_sample} here~~
}
  \item{R}{
%%     ~~Describe \code{R} here~~
}
  \item{message_suppress}{
%%     ~~Describe \code{message_suppress} here~~
}
  \item{\dots}{
%%     ~~Describe \code{\dots} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
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
%%  ~~who you are~~
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
function (evaluation_pt = NULL, posterior_kernel, posterior_sample, 
    R = 50, message_suppress = FALSE, ...) 
{
    if (!is.data.frame(posterior_sample)) 
        stop("The posterior sample needs to be in a data frame")
    if (!is.numeric(R)) 
        stop("The limiting factor needs to be numeric")
    if (!is.function(posterior_kernel)) 
        stop("Invalid posterior kernel function")
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
    n <- nrow(posterior_sample)
    lq <- posterior_kernel(..., xi = evaluation_pt)
    if (is.null(n)) {
        post_diff <- posterior_sample - evaluation_pt
        a <- sin(R * post_diff)/post_diff
        lpostdens <- log(sum(a)) - (log(n) + log(pi))
    }
    else {
        x_mat <- matrix(rep(evaluation_pt, n), nrow = n, byrow = TRUE)
        post_diff <- as.matrix(posterior_sample - x_mat)
        a <- rowProds(sin(R * post_diff)/post_diff)
        lpostdens <- log(sum(a)) - (log(n) + ncol(posterior_sample) * 
            log(pi))
    }
    lq - lpostdens
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
