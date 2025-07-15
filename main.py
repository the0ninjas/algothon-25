import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

currentPos = np.zeros(50)

# --- Utility Functions ---
def get_daily_returns(prices):
    return (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]

def get_volatility(prices, window=20):
    if prices.shape[1] < window + 1:
        window = prices.shape[1] - 1
    returns = get_daily_returns(prices)
    return np.std(returns[:, -window:], axis=1)

def cap_position(prices, positions):
    capped_positions = np.zeros_like(positions)
    for i in range(len(positions)):
        max_shares = int(10000 / prices[i, -1])
        capped_positions[i] = int(np.clip(positions[i], -max_shares, max_shares))
    return capped_positions

# --- Strategy Functions ---
def strategy_cross_sectional_momentum(prices):
    lookback = 30
    if prices.shape[1] < lookback:
        return np.zeros(prices.shape[0])
    returns = (prices[:, -1] - prices[:, -lookback]) / prices[:, -lookback]
    ranks = np.argsort(np.argsort(returns))
    midpoint = len(ranks) // 2
    long = ranks >= midpoint + 10
    short = ranks <= midpoint - 10
    pos = 100 * (long.astype(int) - short.astype(int))
    return cap_position(prices, pos)

def strategy_time_series_momentum(prices):
    lookback = 20
    if prices.shape[1] < lookback:
        return np.zeros(prices.shape[0])
    ma = np.mean(prices[:, -lookback:], axis=1)
    last_price = prices[:, -1]
    signal = last_price - ma
    pos = 100 * np.sign(signal)
    return cap_position(prices, pos)

def strategy_mean_reversion(prices):
    lookback = 10
    if prices.shape[1] < lookback:
        return np.zeros(prices.shape[0])
    ma = np.mean(prices[:, -lookback:], axis=1)
    std = np.std(prices[:, -lookback:], axis=1)
    zscore = (prices[:, -1] - ma) / (std + 1e-8)
    pos = -100 * np.tanh(zscore)
    return cap_position(prices, pos)

def strategy_risk_parity(prices):
    if prices.shape[1] < 2:
        return np.zeros(prices.shape[0])
    vol = get_volatility(prices)
    inv_vol = 1 / (vol + 1e-8)
    weights = inv_vol / np.sum(inv_vol)
    capital = 10000 * len(prices)
    dollar_alloc = weights * capital
    pos = dollar_alloc / prices[:, -1]
    return cap_position(prices, pos)

def strategy_pairs_trading(prices):
    if prices.shape[1] < 50:
        return np.zeros(prices.shape[0])
    stock_a = prices[0, :]
    stock_b = prices[1, :]
    hedge_ratio = np.polyfit(stock_b, stock_a, 1)[0]
    spread = stock_a - hedge_ratio * stock_b
    mean_spread = np.mean(spread[-50:])
    std_spread = np.std(spread[-50:])
    zscore = (spread[-1] - mean_spread) / (std_spread + 1e-8)
    pos = np.zeros(prices.shape[0])
    if zscore > 1:
        pos[0] = -100
        pos[1] = 100 * hedge_ratio
    elif zscore < -1:
        pos[0] = 100
        pos[1] = -100 * hedge_ratio
    return cap_position(prices, pos)

def strategy_pca_mean_reversion(prices, lookback=60, n_components=5):
    if prices.shape[1] < lookback:
        return np.zeros(prices.shape[0])
    returns = np.diff(np.log(prices[:, -lookback:]), axis=1)
    pca = PCA(n_components=n_components)
    pca.fit(returns.T)
    projected = pca.transform(returns.T)
    reconstructed = pca.inverse_transform(projected).T
    residuals = returns - reconstructed
    z_scores = (residuals[:, -1] - residuals.mean(axis=1)) / (residuals.std(axis=1) + 1e-8)
    pos = -100 * np.tanh(z_scores)
    return cap_position(prices, pos)

def strategy_stat_arbitrage(prices, lookback=60):
    if prices.shape[1] < lookback:
        return np.zeros(prices.shape[0])
    returns = np.diff(np.log(prices[:, -lookback:]), axis=1)
    mean_returns = returns.mean(axis=1)
    std_returns = returns.std(axis=1) + 1e-8
    current_return = np.log(prices[:, -1] / prices[:, -2])
    z_scores = (current_return - mean_returns) / std_returns
    pos = -100 * np.tanh(z_scores)
    return cap_position(prices, pos)

def strategy_cluster_mean_reversion(prices, lookback=60, n_clusters=5):
    if prices.shape[1] < lookback:
        return np.zeros(prices.shape[0])
    returns = np.diff(np.log(prices[:, -lookback:]), axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(returns)
    labels = kmeans.labels_
    pos = np.zeros(prices.shape[0])
    for cluster in range(n_clusters):
        indices = np.where(labels == cluster)[0]
        if len(indices) == 0:
            continue
        cluster_returns = returns[indices]
        mean_return = cluster_returns.mean(axis=0)
        std_return = cluster_returns.std(axis=0) + 1e-8
        z_scores = (cluster_returns[:, -1] - mean_return[-1]) / std_return[-1]
        pos[indices] = -100 * np.tanh(z_scores)
    return cap_position(prices, pos)

def strategy_volatility_breakout(prices, lookback=20):
    if prices.shape[1] < lookback + 1:
        return np.zeros(prices.shape[0])
    returns = np.diff(np.log(prices[:, -lookback-1:]), axis=1)
    vol = returns.std(axis=1)
    price_change = prices[:, -1] - prices[:, -2]
    signal = price_change / (vol + 1e-8)
    pos = 100 * np.tanh(signal)
    return cap_position(prices, pos)

def strategy_mean_reverting_portfolio(prices, lookback=60):
    if prices.shape[1] < lookback:
        return np.zeros(prices.shape[0])
    returns = np.diff(np.log(prices[:, -lookback:]), axis=1)
    mean_return = returns.mean(axis=1)
    deviation = returns[:, -1] - mean_return
    pos = -100 * np.tanh(deviation)
    return cap_position(prices, pos)

# --- Main Trading Function ---
def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if nt < 2:
        return np.zeros(nins)

    # Choose one strategy here:
    targetPos = strategy_cross_sectional_momentum(prcSoFar)
    # targetPos = strategy_time_series_momentum(prcSoFar)
    # targetPos = strategy_mean_reversion(prcSoFar)
    # targetPos = strategy_risk_parity(prcSoFar)
    # targetPos = strategy_pairs_trading(prcSoFar)
    # targetPos = strategy_pca_mean_reversion(prcSoFar)
    # targetPos = strategy_stat_arbitrage(prcSoFar)
    # targetPos = strategy_cluster_mean_reversion(prcSoFar)
    # targetPos = strategy_volatility_breakout(prcSoFar)
    # targetPos = strategy_mean_reverting_portfolio(prcSoFar)


    trade = targetPos - currentPos
    trade_value = np.abs(trade * prcSoFar[:, -1])
    # Commission is 5 bps per dollar traded
    cost = 0.0005 * np.sum(trade_value)

    currentPos = targetPos.copy()
    return currentPos
