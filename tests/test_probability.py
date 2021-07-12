import numpy as np
from matplotlib import pyplot as plt

from forward import Heat
from observations import PointObservation
from probability import Prior, Posterior


def test_posterior():
    np.random.seed(134567)
    sig = 0.0005
    N = 400
    time = 5e-2
    L = 2
    alpha = 0.6
    gamma = -0.6


    meas = np.linspace(0, L - 0.2, endpoint=False, num=700) + 0.2
    meas += np.random.normal(scale=0.01, size=meas.size)
    obs = PointObservation(meas=meas, L=L, N=N)
    fwd = Heat(N=N, L=L, alpha=alpha, time=time)
    prior = Prior(gamma=gamma, N=N, L=L)
    prior.multiplier[4:] = 0
    post = Posterior(fwd=fwd, prior=prior, sigSqr=sig ** 2, L=L, N=N)
    u0 = prior.sample(return_coeffs=False)[0]
    uT = fwd(u0)
    data = obs(uT)  # + np.random.normal(scale=sig, size=obs.size)
    post.update(obs, data)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].plot(fwd.x, u0.real, label='IC')
    ax[0].plot(fwd.x, uT.real, label='FC')
    ax[0].scatter(obs.meas, np.dot(post.A, post.to_freq_domain(u0)).real, label='Matrix FC')
    ax[0].scatter(obs.meas, data.real, label='Measurements', marker='*', s=10, color='w', zorder=10)
    line, = ax[0].plot(post.x, post.m, label='Posterior mean')
    # ax[0].plot(post.x, post.m + 2*post.ptwise, color=line.get_color(), label='Posterior std', linestyle=':')
    # ax[0].plot(post.x, post.m - 2*post.ptwise, color=line.get_color(), linestyle=':')
    ax[0].legend()
    ax[0].set_title("Error bars seem too small, no?")

    ax[1].plot(post.x, post.ptwise.real, label='posterior STD')
    ax[1].scatter(obs.meas, np.zeros(obs.size), label='measurements')
    # print(np.diag(post.Sigma)[:9])
    # tra = post.to_freq_domain(post.m)
    # plt.close()
    # plt.plot(tra)