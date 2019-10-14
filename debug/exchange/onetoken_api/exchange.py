"""
这个demo包含一些公开的接口
"""
from pprint import pprint
import requests
import pandas as pd
import time
from .demo_private import api_call
import threading

class OneToken():
    def __init__(self,debug = False):
        self.debug = debug

        try:
            self.exchanges = pd.read_csv( 'exchanges.csv' )
        except:
            self.exchanges = self.get_exchanges()

        self.tickets = None

    def get_time(self):
        res = requests.get('https://1token.trade/api/v1/basic/time')
        pprint(res.json())

    def get_exchanges(self):
        # 获取交易所信息
        res = requests.get('https://1token.trade/api/v1/basic/support-exchanges-v2')
        # pprint(res.json(), width=240)
        exchanges = pd.DataFrame(res.json(), columns=['exchange', 'alias', 'sub_markets', 'sub_markets_alias', 'type'])
        #print(exchanges)
        exchanges.to_csv('exchanges.csv')
        return exchanges

class Exchange():
    # API method metainfo
    has = {
        'cancelAllOrders': True,
        'cancelOrder': True,
        'cancelOrders': True,
        'CORS': False,
        'createDepositAddress': False,
        'createLimitOrder': True,
        'createMarketOrder': True,
        'createOrder': True,
        'deposit': False,
        'editOrder': 'emulated',
        'fetchBalance': True,
        'fetchClosedOrders': False,
        'fetchCurrencies': False,
        'fetchDepositAddress': False,
        'fetchDeposits': False,
        'fetchL2OrderBook': True,
        'fetchLedger': False,
        'fetchMarkets': True,
        'fetchMyTrades': False,
        'fetchOHLCV': 'emulated',
        'fetchOpenOrders': False,
        'fetchOrder': False,
        'fetchOrderBook': True,
        'fetchOrderBooks': False,
        'fetchOrders': False,
        'fetchStatus': 'emulated',
        'fetchTicker': True,
        'fetchTickers': False,
        'fetchTime': False,
        'fetchTrades': True,
        'fetchTradingFee': False,
        'fetchTradingFees': False,
        'fetchFundingFee': False,
        'fetchFundingFees': False,
        'fetchTradingLimits': False,
        'fetchTransactions': False,
        'fetchWithdrawals': False,
        'privateAPI': True,
        'publicAPI': True,
        'withdraw': False,
    }

    def __init__(self,account = None):
        self.account = account

        self.exchange = self.account.split('/')[0]

        self._contract_file = '{}_contract.csv'.format(self.exchange)

        self.balance = None
        self.position = None
        self.orders = None

        try:
            self.contract = pd.read_csv(self._contract_file)
        except FileNotFoundError:
            self.contract = self.get_contract()

        self.get_balance()

        self.get_quote_tickets()

    def get_contract(self):
        res = requests.get( 'https://1token.trade/api/v1/basic/contracts?exchange={}'.format( self.exchange ) )
        # pprint(res.json(), width=1000)
        df = pd.DataFrame( res.json() )  #
        #df.to_csv(self._contract_file)
        #print('save {} done'.format(self._contract_file))
        return df

    def get_quote_tickets_threads(self):
        t = threading.Thread(target= self.get_tickets_all_the_time,args=[])
        t.setDaemon(True)
        t.start()

    def get_tickets_all_the_time(self):
        while True:
            self.get_quote_tickets()

    def get_quote_tickets(self):
        res = requests.get( 'https://1token.trade/api/v1/quote/ticks?exchange={}'.format( self.exchange ) )
        # pprint(res.json()[:3], width=1000)
        r = res.json()

        try:
            tickets = pd.DataFrame(r)
            tickets['exchange'] = self.exchange
            tickets['contract'] = list( map( lambda x: x.split( '/' )[1], tickets['contract'] ) )
            tickets['ask_price'] = list( map( lambda x: x[0]['price'], tickets['asks'] ) )
            tickets['ask_volume'] = list( map( lambda x: x[0]['volume'], tickets['asks'] ) )
            tickets['bid_price'] = list( map( lambda x: x[0]['price'], tickets['bids'] ) )
            tickets['bid_volume'] = list( map( lambda x: x[0]['volume'], tickets['bids'] ) )
            del tickets['asks']
            del tickets['bids']

            self.tickets = tickets

            #print(tickets)
            return tickets

        except:
            print('get_quote_tickets',res.json())

    def get_single_ticket(self,contract):
        res = requests.get('https://1token.trade/api/v1/quote/single-tick/{}/{}'.format(self.exchange,contract))
        ticket = pd.DataFrame(res.json(), columns=['contract', 'last', 'asks', 'bids'])
        try:
            ticket['ask_price'] = ticket['asks'][0]['price']
            ticket['ask_volume'] = ticket['asks'][0]['volume']
            ticket['bid_price'] = ticket['bids'][0]['price']
            ticket['bid_volume'] = ticket['bids'][0]['volume']
            del ticket['asks']
            del ticket['bids']
        except:
            print(res.json())
        
        return ticket

    # def get_full_ticks(self):
    #     url = 'https://hist-quote.1tokentrade.cn/ticks/full?date={}&contract={}'.format(date, contract)

    def trade(self,contract,type,price,amount,client_oid = None):
        r = api_call( 'POST', '/{}/orders'.format( self.account ),
                      data={'contract': contract, 'price': price, 'client_oid': client_oid, 'bs': type, 'amount': amount,
                            'options': {'close': False}} )

        # print(r.json())
        return r.json()

    def buy(self,contract,price,amount):
        r = api_call('POST', '/{}/orders'.format(self.account),
                     data={'contract': contract, 'price': price, 'bs': 'b', 'amount': amount})

        #print(r.json())
        return r.json()

    def sell(self,contract,price,amount):
        r = api_call('POST', '/{}/orders'.format(self.account),
                     data={'contract':contract, 'price': price, 'bs': 's', 'amount': amount})

        return r.json()

    def get_balance(self):
        r = api_call('GET', '/{}/info'.format(self.account))
        balance = r.json()

        position = balance['position']
        pos_df = pd.DataFrame( position )
        pos_df.set_index(["contract"], inplace=True)

        self.balance = balance
        self.position = pos_df


    def close(self):
        r = api_call('DELETE', '/{}/orders/all'.format(self.account))
        return r.json()

    def get_orders(self,exg_oid=None):
        r = api_call('GET', '/{}/orders'.format(self.account), params={'exchange_oid': exg_oid})
        #pprint(r.json(), width=100)
        try:
            self.orders = pd.DataFrame(r.json())
        except:
            print(r.json())

        return r.json()

    def close_order(self,exg_oid):
        r = api_call('DELETE', '/{}/orders'.format(self.account), params={'exchange_oid': exg_oid})
        return r.json()

    def get_trans(self,contract):
        r = api_call('GET','/{}/trans'.format(self.account),params={'contract': contract})
        return r.json()

def demo_onetoken():
    onetoken = OneToken()

    onetoken.get_time()
    # 获取支持的交易所
    print(onetoken.exchanges)


def demo_exchange():
    okex = Exchange('okex/mock-luyh-okex')

    balance = okex.balance()
    position = okex.position
    position['btc']['available']
    print('position',okex.position)



    exg_oid = okex.buy('btc.usdt',1000,0.1)
    print('用 exchange oid撤单')
    r = okex.close_order(exg_oid)
    pprint(r, width=100)


    print('查询挂单 应该没有挂单')
    orders = okex.get_orders()
    pprint(orders, width=100)
    assert len(orders) == 0

    print('okex_contract', okex.contract)
    okex_tickets = okex.get_quote_tickets()
    print('okex_tickets', okex_tickets)

    okef = Exchange('okef')
    okef_btc_usd_q_ticket = okef.get_single_ticket('btc.usd.q')
    print('okef_btc_usd_q_ticket', okef_btc_usd_q_ticket)



if __name__ == '__main__':
    #demo_onetoken()

    demo_exchange()

    print('end')

