from constraints.stcd.linear import Linear


def mining_stcd(mts, win_size=2, confidence=0.95, n_components=1, max_len=3,
                sample_train=1.0, pc=1.0, ridge_all=False, implication_check=False,
                max_x=1, verbose=0):
    stcd = Linear(mts, win_size, confidence, n_components, max_len,
                  sample_train, pc, ridge_all, implication_check)
    stcd.mine(max_x=max_x, verbose=verbose)
    return stcd.rules
