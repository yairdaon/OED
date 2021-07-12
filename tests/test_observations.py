from observations import PointObservation


def test_point_observation():
    k = 7
    obs = PointObservation(N=1400, L=2, meas=np.linspace(0,2,50))
    u = obs.eigenvector(k)

    fig = plt.figure(figsize=(6,3))
    plt.plot(obs.x, u)
    plt.scatter(obs.meas, obs(u).real)
    plt.show()