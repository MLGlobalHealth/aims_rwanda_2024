data {
    int<lower=1> N;                 // number of observations
    int<lower=1> K;
    vector[N] y;                    // outcome variable
    matrix[N,K] X;                  // design matrix for fixed effects

    // data for GP
    int<lower=1> NI;                // number of unique inputs
    array [NI] real inputs_standardised; // unique inputs for GP
    array [N] int<lower=1, upper=NI> map_unique_inputs_to_obs;
}

transformed data {
  real gp_nugget = 1e-9; // GP nugget
}

parameters {
    real<lower=0> sigma;
    real beta0;
    vector[K] beta;

    real<lower=0> gp_lengthscale;
    real<lower=0> gp_sigma;

    vector[NI] z;
}

transformed parameters{
    vector[N] mu;
    vector[NI] f;

    {
      matrix[NI, NI] K_f;
      matrix[NI, NI] L_f;

      // compute the Exponentiated quadratic covariance function
      K_f = gp_exp_quad_cov(inputs_standardised, gp_sigma, gp_lengthscale);
      // compute the Cholesky decomposition
      L_f = cholesky_decompose(add_diag(K_f, gp_nugget));
      // sample GP at inputs
      f = L_f * z;
    }

    mu = beta0 + X * beta + f[map_unique_inputs_to_obs];
}

model {
    // priors for observation noise
    sigma ~ cauchy( 0 , 1 );

    // priors for baseline and fixed effects
    beta0 ~ normal( 0 , 10 );
    beta ~ normal( 0 , 1 );

    // priors for GP
    gp_lengthscale ~ inv_gamma( 5, 1 );
    gp_sigma ~ cauchy( 0 , 1 );
    z ~ std_normal();

    // likelihood
    y ~ normal( mu , sigma );
}
