# Script to plot logistic and gompertz fits to observations diseased people in a variety of countries
# Fits are performed with non-linear least squares and plotted in time using Runge-Kutta using the 
# Ordinary differential equations for both logistic and Gompertz.

import requests
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from numpy import log
import scipy.optimize as spo
from scipy import integrate

# Define some convenience functions
def sird(x, gamma=1, eta=1, beta=1):
    return gamma*(1 - np.exp(- eta * x)) - beta*x
def sird_jac(x, gamma, eta, beta):
    J = np.empty((x.size, 3))
    J[:, 0] = 1 - np.exp(-eta*x)
    J[:, 1] = gamma*x*np.exp(-eta*x)
    J[:, 2] = -x
    return J
def logistic(x, b, k):
    return b * x * (1 - x / k)

def get_b_k(gamma, eta, beta):
    b = gamma*eta - beta
    k = gamma * eta**2/(2*b)
    return b, k

def d_small(x, gamma=1, eta=1, beta=1):
    b,k = get_b_k(gamma, eta, beta)
    return b*x*(1-k*x)

def gomp(x, beta=1, alpha=1, x_tilde=1):
    return beta*x - alpha*x*np.log((x + 0.0001) / x_tilde)

def gomp_jac(x, beta, alpha, x_tilde):
    J = np.empty((x.size, 3))
    J[:, 0] = x
    J[:, 1] = -x*np.log(x/x_tilde)
    J[:, 2] = alpha*x/x_tilde
    return J

def get_gomp_inf(beta, alpha, x_tilde):
    return x_tilde * np.exp(beta / alpha)

def gomp_time(t, x0, beta, alpha, x_tilde):
    xinf = get_gomp_inf(beta, alpha, x_tilde)
    gamma = x0 / xinf
    return xinf * np.power(gamma, np.exp(- alpha * t))

def gomp_norm(lns, alpha):
    return - alpha * lns

def get_transmission_rate(gamma, eta, cov):
    # source: https://en.wikipedia.org/wiki/Distribution_of_the_product_of_two_random_variables#Correlated_central-normal_distributions
    # cov is in order gamma, eta, beta
    # from Carletti's definition and the SIRD here
    # returns rate and standard error
    transmission_rate = eta * gamma
    rho = cov[0, 1]
    gamma_var = cov[0, 0]
    eta_var = cov[1, 1]
    var_transmission = gamma_var * eta_var * (1 + rho**2) + eta_var*(gamma**2) + gamma_var * (eta**2) 
    return transmission_rate, np.sqrt(var_transmission)

def get_R_0(gamma, eta, beta, cov):
    # cov is in order gamma, eta, beta
    # from Carletti's definition and the SIRD here
    # returns rate and standard error in the form of 95th percent confidence interval
    # Ignore negative R_0 values on physical grounds
    # R_0 confidence interval by bootstrap
    B = 10000
    R_0 = np.zeros(B)
    x = np.random.multivariate_normal(mean=np.array([gamma, eta, beta]), cov=cov, size=B)
    for i, xrow in enumerate(x):
        R_0[i] = xrow[0] * xrow[1] / xrow[2]
        if R_0[i] < 0:
            R_0[i] = 0

    R_0[R_0 == 0] = np.mean(R_0[R_0 != 0])
    
    if np.quantile(R_0, 0.975) - np.quantile(R_0, 0.025) > 10:
        print(np.quantile(R_0, 0.975) - np.quantile(R_0, 0.025))
    return gamma * eta / beta, np.quantile(R_0, 0.975) - np.quantile(R_0, 0.025)
    
def gomp_fit(t, yraw):
    y = np.copy(yraw)
    yinf = np.max(y)
    y = y / yinf
    lly = safe_lnln(y)
    interp = np.ones(len(lly))
    np.linalg.lstsq()
    

def sirdODE(t, y, r, a, d, s0):
    S, I, R, D = y
    Sp = -r/s0 * S * I
    Ip = r/s0 * S * I - (a + d) * I
    Rp = a * I
    Dp = d * I
    yp = Sp, Ip, Rp, Dp
    return yp

def sird_d_ODE(t, y, gamma=1, eta=1, beta=1):
    # Deaths only
    yp = sird(y, gamma, eta, beta)
    return yp

def sird_dsmall_ODE(t, y, gamma=1, eta=1, beta=1):
    # Deaths only
    yp = d_small(y, gamma, eta, beta)
    return yp

def safe_lnln(x, k=1):
    z = x / k
    eps = 5e-3
    z[z == 0] = eps
    z[z >= 1.] = 1 - eps
    return np.log(-np.log(z))

def sird_d_time(t, x0, gamma, eta, beta, norm=False):
    tmax = np.max(t)
    tmin = np.min(t)
    obj = integrate.solve_ivp(sird_d_ODE, (tmin, tmax), (x0,), 
                              t_eval=np.arange(tmin, tmax + 1), 
                              args=(gamma, eta, beta),)
    d_sird = obj["y"].flatten()
    if norm:
        d_sird = d_sird / d_sird.max()
    return d_sird

def sird_dsmall_time(t, x0, gamma, eta, beta, norm=False):
    tmax = np.max(t)
    
    obj = integrate.solve_ivp(sird_dsmall_ODE, (0, tmax), (x0,), 
                              t_eval=np.arange(tmax + 1), args=(gamma, eta, beta))
    d_sird = obj["y"].flatten()
    if norm:
        d_sird = d_sird / d_sird.max()
    return d_sird

def logistic_time(t, x0, b, k):
    tmax = np.max(t)
    A = (k - x0) / x0
    denom =1 + A * np.exp(-b*t) 

    return k/denom

def sird_time(t, dx0, x0, mu, gamma, eta, beta, xinf=1, norm=False):
    # mu is recovery / death ratio
    tmax = np.max(t)
    d, r = mu * beta, eta * gamma
    s0 = gamma / d
    
    i0 = dx0 / d
    r0 = mu * x0
    a = beta - d
    y0 = s0, i0, r0, x0
    print(f"d: {d:.3f},a:{a:.3f},r:{r:.2f}")
#     print(f"Implied s0: {s0:.3f}")
#     print(f"initial slope: {gamma * eta - beta}")
    obj = integrate.solve_ivp(sirdODE, (0, tmax), y0, 
                              t_eval=np.arange(tmax + 1), args=(r, a, d, s0))
    d_sird = obj["y"][3, :]
#     from pdb import set_trace
#     set_trace()
    if norm:
        d_sird = obj["y"][3, :] / obj["y"][3, :].max()
    return d_sird, d, a, r, obj

START_DATE = "2020-01-22"
END_DATE = "2020-05-16"
WINDOW = 6 # days

# Primary data source
def get_data2(country, window=WINDOW):
    y = data2[data2["Country/Region"] == country].iloc[:, 4:].sum(axis=0)
    dates = pd.to_datetime(data2.columns[4:], format="%m/%d/%y")
    cumdeaths = pd.Series(index=dates[1:], data=y.values.flatten()[1:])
    deaths = pd.Series(index=dates[1:], data=np.diff(y.values.flatten()))
    if country == "China":
        deaths = deaths[~((deaths > 1000))]
    cumdeaths = deaths.cumsum()
    deaths = deaths.rolling(WINDOW, center=True).mean()
    country = pd.DataFrame(index=dates[1:], data={"rollDeaths": deaths, "cumDeaths": cumdeaths})
    country = country[START_DATE:END_DATE]
    country = country.dropna()
    return country

# Alternative data source (obsolete)
def get_data(country: str):

    country = data[data.countriesAndTerritories == country]
    country.index = pd.to_datetime(country.dateRep, format="%d/%m/%Y")
    country = country.sort_index()
    country = country[START_DATE:END_DATE]
    country["rollDeaths"] = country.deaths.rolling(WINDOW, center=True).mean()
    country["cumDeaths"] = country.rollDeaths.cumsum()
    country["rollDeathsNorm"] = country["rollDeaths"] / country["cumDeaths"].max()
    country["cumDeathsNorm"] = country["cumDeaths"] / country["cumDeaths"].max()
    country = country.dropna()
    return country

# Conversion of numbers and errors to strings with uncertainty 
# e.g. value=1.002, error=0.3 => 1.0(3)
def conv2siunitx(val, err, err_points=1, si=True):
    value = f'{val:.20e}'.split('e')
    error = f'{err:.20e}'.split('e') 
    val, val_exp = float(value[0]), int(value[1])
    err, err_exp = float(error[0]), int(error[1])
    diff = val_exp - err_exp
    my_val = f'{np.round(val, diff):.10f}'
    my_err = f'{np.round(err, err_points-1):.10f}'.replace('.','')

    if not si:
        c = np.power(10, float(val_exp))
        num = np.round(val, diff) * c
        my_val = f'{num:.10f}'
        
        if err_exp > 0: # Account for more digits in errors > 10
            err_points += err_exp

        if err_exp >= 0 and val_exp > 0:
            first_uncertain = abs(val_exp) + 1
        if err_exp < 0 and val_exp < 0:
            first_uncertain = abs(val_exp) + 2 + diff
        if err_exp < 0 and val_exp > 0:
            first_uncertain = val_exp + abs(err_exp) + 2
        if val_exp == 0 and err_exp < 0:
            first_uncertain = diff + 2
        if err_exp >= 0 and val_exp == 0:
            first_uncertain = 1
        val_exp = ""
    else:
        val_exp = f"e{val_exp}"            
        if val_exp == "e+00":
            val_exp = ''
            
    return(f'{my_val[:first_uncertain]}({my_err[:err_points]}){val_exp}')


assert conv2siunitx(980, 70, si=False) == "980(70)"
assert conv2siunitx(980, 73, si=False) == "980(70)"
assert conv2siunitx(9802, 731, si=False) == "9800(700)"
assert conv2siunitx(0.02, 0.003, si=False) == "0.020(3)"
assert conv2siunitx(1.0321, 0.0003, si=False) == '1.0321(3)'
assert conv2siunitx(9802, 0.731, si=False) == "9802.0(7)"
assert conv2siunitx(2, 1, si=False) == "2(1)"
assert conv2siunitx(937.697300, 80, si=False) == "940(80)"

