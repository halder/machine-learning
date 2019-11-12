def create_data(n_range):
    temp_outside = list()
    temp_inside = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for _ in range(n_range):
        pp = (1-0.5)/19
        n = np.random.choice(range(1,21), p=[pp,pp,pp,pp,pp,pp,pp,pp,pp,pp,pp,pp,pp,pp,pp,pp,pp,pp,pp, 1-(pp*19)])
        temp = list()
        for i, L in enumerate(temp_inside):
            if i < n:
                temp.append(1)
            else:
                temp.append(L)
        temp_outside.append(temp)

    data = np.array(uppertemp)
    X = data[:, :-1]
    y = data[:, -1]
    
    return X, y
