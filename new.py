from typing import Dict, List, Optional, Tuple, Callable
import os, sys, time, requests, signal
import math
from typing import NamedTuple
import requests
from scipy.stats import norm


API_KEY = os.getenv("RIT_API_KEY", "YOUR_API_KEY_HERE")  # Replace with your actual API key
BASE_URL = "http://localhost:9999/v1"

s = requests.Session()
s.headers.update({"X-API-key": API_KEY})

#Current Stock price
def get_last(ticker: str) -> float:
    resp = s.get('http://localhost:9999/v1/securities', params={'ticker': ticker})
    if resp.ok:
        securities = resp.json()
        if securities and len(securities) > 0:
            return float(securities[0]['last'])
    return 0.0
# last_price = get_last(SECURITIES.CALLS.RTM48C)


#Time remaining in RIT
def get_time_remaining() -> Tuple[float, float]:
#check with case
    resp = s.get('http://localhost:9999/v1/case')
    if resp.ok:
        case = resp.json()
        # Get seconds remaining out of 300 total seconds
        seconds_remaining = float(case.get('time_remaining', 0))
        total_seconds = 300.0
        
        # Calculate days remaining (like =ROUNDUP(B7/15,0) in Excel)
        # where B7 is time remaining in seconds divided by 15 seconds per day
        days_remaining = math.ceil(seconds_remaining / 15)
        
        # Calculate years remaining (like =B12/240 in Excel)
        # where B12 is days remaining divided by 240 trading days per year
        years_remaining = days_remaining / 240
        
        return days_remaining, years_remaining
    return 0.0, 0.0

def get_news():
    resp = s.get('http://localhost:9999/v1/news')
    if resp.ok:
        news = resp.json()
        news_id = news[0]['news_id']
        headline = news[0]['headline']
        body = news[0]['body']
        return news_id, headline, body


#get news function that will update vol based on news
def get_vol():
    news = get_news()
    news_id = news[0]
    if news_id == 0: # no news, use initial value
        return 20.0
    else:
        
        extract vol


class SECURITIES:
    ETF = "RTM"
    
    class CALLS:
        RTM48C = "RTM48C"
        RTM49C = "RTM49C"
        RTM50C = "RTM50C"
        RTM51C = "RTM51C"
        RTM52C = "RTM52C"
        
    class PUTS:
        RTM48P = "RTM48P"
        RTM49P = "RTM49P"
        RTM50P = "RTM50P"
        RTM51P = "RTM51P"
        RTM52P = "RTM52P"
    
    # Dictionary for strike prices
    STRIKES = {
        CALLS.RTM48C: 48, PUTS.RTM48P: 48,
        CALLS.RTM49C: 49, PUTS.RTM49P: 49,
        CALLS.RTM50C: 50, PUTS.RTM50P: 50,
        CALLS.RTM51C: 51, PUTS.RTM51P: 51,
        CALLS.RTM52C: 52, PUTS.RTM52P: 52
    }


class PositionLimits:
    ETF_MAX_ABS = 50000
    OPT_GROSS_LIMIT = 2500
    OPT_NET_LIMIT = 1000

    #can you double check

class BSParameters:
    S: float = get_last("RTM")  # Underlying price
    r: float = 0  # Risk-free rate
    q: float = 0.0   # Dividend yield
    T: float = get_time_remaining()[1]   # Time to expiry (years)
    sigma: float = get_vol()  # Volatility


class PositionTracker:
    def __init__(self, client):
        self.client = client
        self.positions = {}
        self.update_positions()

    
    def get_position(self, ticker: str) -> int:
        return self.positions.get(ticker, 0)
    
    def get_total_positions(self) -> Tuple[int, int]: 
        #this is only Options total position!
        gross = 0
        net = 0
        for ticker, pos in self.positions.items():
            if ticker != SECURITIES.ETF:  
                gross += abs(pos)
                net += pos
        return gross, net
    
    #condition met 
    def check_limits(self, new_orders: Dict[str, int]) -> bool:
        test_positions = self.positions.copy()
        
        # Add new orders
        for ticker, qty in new_orders.items():
            test_positions[ticker] = test_positions.get(ticker, 0) + qty
        
        # Check ETF limit
        if abs(test_positions.get(SECURITIES.ETF, 0)) > PositionLimits.ETF_MAX_ABS:
            return False
            
        # Calculate new option totals
        gross = 0
        net = 0
        for ticker, pos in test_positions.items():
            if ticker != SECURITIES.ETF:
                gross += abs(pos)
                net += pos
                
        # Check option limits
        if gross > PositionLimits.OPT_GROSS_LIMIT:
            return False
        if abs(net) > PositionLimits.OPT_NET_LIMIT:
            return False
            
        return True

####################################
def bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if is_call:
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (math.log(S/K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T)


def implied_volatility(option_price, S, K, T, r, option_type='call', tol=1e-6, max_iter=100):
    sigma = BSParameters.sigma # initial guess is benchmark vol
    
    for i in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        v = vega(S, K, T, r, sigma)
        
        price_diff = price - option_price
        
        if abs(price_diff) < tol:
            return sigma
        
        sigma -= price_diff / v  # Newton-Raphson update
    
    return None  # if it doesn't converge


def delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    d1 = (math.log(S/K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if is_call:
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1



def hedge():



# # Example usage: hedge ratio??
# def get_hedge_ratio(option_ticker: str) -> float:

#     # Get current market data

#     #you only check delta when you have open position

#     S = get_last(SECURITIES.ETF)
#     K = SECURITIES.STRIKES[option_ticker]
#     _, T = get_time_remaining()
#     r = 0.0  # risk-free rate assumption
#     is_call = option_ticker in vars(SECURITIES.CALLS)
#     market_price = get_last(option_ticker)
#     implied_vol = find_implied_vol_scipy(market_price, S, K, T, r, is_call)
#     delta = calculate_delta(S, K, T, r, implied_vol, is_call)
    
#     return delta


def main():
# 1. get IV from the book options
# 2. compare to BS.sigma (benchmark)
# 3. if good enough, trade it
# 4. delta hedge
# 4. ensure delta is within range





#BRO what is OOP
#FUCKKKKOFF stupid piece of shti

#buy order / submit order:     
# resp = s.post('http://localhost:9999/v1/orders', params = {'ticker': ticker_symbol2, 'type': 'LIMIT', 'quantity': order_quantity, 'price': best_bid_price + buffer, 'action': 'BUY'})

