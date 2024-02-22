functions {
  vector diagSPD_EQ(real alpha, real rho, real L, int M)
  {
    return alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
  }

  matrix PHI(int N, int M, real L, vector x)
  {
    return sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
  }
}

data{
    int<lower=1> N;                 // number of observations
    int<lower=1> K;
    vector[N] y;                    // outcome variable
    matrix[N,K] X;                  // design matrix for fixed effects

    // data for GP
    int<lower=1> NI;                // number of unique inputs
    vector[NI] inputs_standardised; // unique inputs for GP
    array [N] int<lower=1, upper=NI> map_unique_inputs_to_obs;

    // HSGP arguments
    real<lower=0> hsgp_c;   // factor c to determine the boundary value L for the HSGP
    int<lower=1> hsgp_M;    // number of basis functions for the HSGP
}
transformed data
{
    matrix[NI, hsgp_M] hsgp_PHI;
    real hsgp_L;

    // precompute HSGP basis functions at inputs
    hsgp_L = hsgp_c*max(inputs_standardised);
    hsgp_PHI = PHI(NI, hsgp_M, hsgp_L, inputs_standardised);
}
parameters{
    real<lower=0> sigma;
    real beta0;
    vector[K] beta;

    real<lower=0> gp_lengthscale;
    real<lower=0> gp_sigma;

    vector[hsgp_M] z;
}
transformed parameters{
    vector[N] mu;
    vector[NI] f;

    {
      //declare transformed parameters in local scope that won't appear in Stan output
      vector[hsgp_M] hsgp_sqrt_spd;

      // square root of spectral densities
      hsgp_sqrt_spd = diagSPD_EQ( gp_sigma, gp_lengthscale, hsgp_L, hsgp_M);
      // construct HSGP at inputs
      f = hsgp_PHI * (hsgp_sqrt_spd .* z);
    }

    mu = beta0 + X * beta + f[map_unique_inputs_to_obs];
}

model{
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

generated quantities {
  array[N] real yhat; // Generate predictions
  for (i in 1:N) {
    yhat[i] = normal_rng(mu[i], sigma);
  }
}
