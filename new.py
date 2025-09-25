from typing import Dict, List, Optional, Tuple, Callable
import os, sys, time, requests, signal
import math
from typing import NamedTuple
import requests

from scipy.stats import norm
from scipy.optimize import root_scalar

API_KEY = os.getenv("RIT_API_KEY", "YOUR_API_KEY_HERE")  # Replace with your actual API key
BASE_URL = "http://localhost:9999/v1"

s = requests.Session()
s.headers.update({"X-API-key": API_KEY})




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
    
    # ALL = [ETF,
    #        CALLS.RTM48C, PUTS.RTM48P,
    #        CALLS.RTM49C, PUTS.RTM49P,
    #        CALLS.RTM50C, PUTS.RTM50P,
    #        CALLS.RTM51C, PUTS.RTM51P,
    #        CALLS.RTM52C, PUTS.RTM52P]
    
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
    def __init__(self):
        self.S: float = 0.0  # Underlying price
        self.r: float = 0  # Risk-free rate
        self.q: float = 0.0   # Dividend yield
        self.T: float = 0.0   # Time to expiry
        self.sigma: float = 0.20  # Volatility


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


#back out IV

#THIS IS WRONG CHECK BSSSSS actual bs


def bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    
    #T = years_remaining thus float
    #r= 0 assumption

    d1 = (math.log(S/K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    

    #theo prices !!!!!!!!!
    if is_call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)



def find_implied_vol_scipy(target_price: float, S: float, K: float, T: float, r: float, is_call: bool) -> float:
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, is_call) - target_price
        
    result = root_scalar(objective, 
                        bracket=[0.0001, 5.0],
                        method='brentq') #confused..
    return result.root if result.converged else 0.20  # Return benchmark vol if no convergence


def calculate_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0:
        return 1.0 if is_call else -1.0  # At expiry
        
    d1 = (math.log(S/K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    
    if is_call:
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1  # or -norm.cdf(-d1)

# Example usage: hedge ratio??
def get_hedge_ratio(option_ticker: str) -> float:

    # Get current market data

    #you only check delta when you have open position

    S = get_last(SECURITIES.ETF)
    K = SECURITIES.STRIKES[option_ticker]
    _, T = get_time_remaining()
    r = 0.0  # risk-free rate assumption
    is_call = option_ticker in vars(SECURITIES.CALLS)
    market_price = get_last(option_ticker)
    implied_vol = find_implied_vol_scipy(market_price, S, K, T, r, is_call)
    delta = calculate_delta(S, K, T, r, implied_vol, is_call)
    
    return delta






class BSOut(NamedTuple):
    d1: float
    d2: float
    theo: float







#BRO what is OOP
#FUCKKKKOFF stupid piece of shti

#buy order / submit order:     
# resp = s.post('http://localhost:9999/v1/orders', params = {'ticker': ticker_symbol2, 'type': 'LIMIT', 'quantity': order_quantity, 'price': best_bid_price + buffer, 'action': 'BUY'})

