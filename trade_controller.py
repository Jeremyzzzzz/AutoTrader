from binance_client import get_client
import logging
from decimal import Decimal, ROUND_DOWN

client = get_client()

def cancel_all_orders(symbol):
    """取消指定交易对的所有订单"""
    try:
        client.futures_cancel_all_open_orders(symbol=symbol)
        print(f"√ 已取消{symbol}所有订单")
    except Exception as e:
        logging.error(f"取消订单失败: {e}")

def check_position():
    """
    检查账户是否持有任何合约仓位
    :return: 如果有任何未平仓仓位返回 True，否则返回 False
    """
    try:
        account_info = client.futures_account()  # 获取合约账户信息
        positions = account_info['positions']

        for position in positions:
            position_amt = float(position['positionAmt'])  # 持仓数量
            if position_amt != 0:
                return True

        return False

    except Exception as e:
        logging.error(f"检查账户持仓时发生错误: {e}")
        return False

def close_existing_position(symbol):
    """ 平掉当前持仓并取消关联订单 """
    try:
        # 先取消所有订单
        cancel_all_orders(symbol)  # 新增：取消所有未完成订单
        
        # 获取当前持仓
        positions = client.futures_position_information()
        for position in positions:
            if position['symbol'] == symbol and float(position['positionAmt']) != 0:
                side = 'SELL' if float(position['positionAmt']) > 0 else 'BUY'
                quantity = abs(float(position['positionAmt']))
                
                # 市价平仓
                client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
                print(f"已平仓: {quantity} {symbol}")
                
                # 再次取消订单确保没有残留订单（新增）
                cancel_all_orders(symbol)
                return True
        return False
    except Exception as e:
        logging.error(f"平仓失败: {e}")
        return False

def check_balance():
    """ 查询合约钱包余额 """
    try:
        balances = client.futures_account_balance()
        for balance in balances:
            if balance['asset'] == 'USDT':
                return float(balance['balance'])
    except Exception as e:
        logging.error(f"查询余额失败: {e}")
        return None



def close_all_positions():
    """平掉所有未平仓的合约仓位"""
    try:
        # 获取账户持仓信息
        positions = client.futures_position_information()

        for position in positions:
            position_amt = float(position['positionAmt'])  # 持仓数量
            symbol = position['symbol']

            if position_amt != 0:  # 如果持仓数量不为 0
                side = 'SELL' if position_amt > 0 else 'BUY'  # 根据持仓方向决定平仓方向
                quantity = abs(position_amt)  # 获取平仓数量（绝对值）

                # 提交市场订单平仓
                client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
                print(f"已平仓: {quantity} {symbol}")

    except Exception as e:
        logging.error(f"平仓失败: {e}")



def get_symbol_precision(symbol):
    """获取交易对的精度限制"""
    try:
        exchange_info = client.get_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                quantity_precision = s['quantityPrecision']
                price_precision = s['pricePrecision']
                print(f"交易对 {symbol} 的数量精度: {quantity_precision}, 价格精度: {price_precision}")
                return quantity_precision, price_precision
    except Exception as e:
        logging.error(f"获取交易对 {symbol} 精度失败: {e}")
        raise ValueError(f"无法获取交易对 {symbol} 的精度信息")


def adjust_precision(value, precision):
    """调整小数位数，确保符合精度限制"""
    value = Decimal(value)
    return float(value.quantize(Decimal(f"1e-{precision}"), rounding=ROUND_DOWN))


def get_trade_precision(symbol):
    """
    获取交易对的最小交易量步长 (stepSize)
    """
    try:
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                step_size = float(next(f['stepSize'] for f in s['filters'] if f['filterType'] == 'LOT_SIZE'))
                return step_size
    except Exception as e:
        logging.error(f"获取交易对 {symbol} 精度失败: {e}")
        return None


def get_price_precision(symbol):
    """
    获取交易对的价格精度 (tickSize)
    """
    try:
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                price_filter = next((f for f in s['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if price_filter:
                    return float(price_filter['tickSize'])
                else:
                    raise ValueError(f"{symbol} 没有找到 PRICE_FILTER 过滤器")
    except Exception as e:
        logging.error(f"获取交易对 {symbol} 的价格精度失败: {e}")
        return None


def adjust_price(price, tick_size):
    """
    调整价格为符合精度的值
    """
    if tick_size is None:
        raise ValueError("tick_size 为 None，无法调整价格")
    
    tick_size_str = f"{tick_size:.10f}".rstrip('0')
    if '.' in tick_size_str:
        precision = len(tick_size_str.split('.')[1])
    else:
        precision = 0

    return round(price // tick_size * tick_size, precision)


def place_order(symbol, side, quantity, price, stop_loss, take_profit):
    """
    下单并设置止损和止盈，使用限价单
    :param symbol: 交易对，例如 BTCUSDT
    :param side: 交易方向 'BUY' 或 'SELL'
    :param quantity: 订单数量
    :param price: 限价下单的价格
    :param stop_loss: 止损价格
    :param take_profit: 止盈价格
    """
    try:
        # 获取交易对的价格精度
        tick_size = get_price_precision(symbol)
        if tick_size is None:
            raise ValueError(f"无法获取交易对 {symbol} 的 tick_size")

        # 调整价格精度
        price = adjust_price(price, tick_size)
        stop_loss = adjust_price(stop_loss, tick_size)
        take_profit = adjust_price(take_profit, tick_size)

        tick_size = get_trade_precision(symbol)

        # 计算 tick_size 小数点后位数
        if tick_size is not None:
            tick_size_str = f"{tick_size:.10f}".rstrip('0')  # 转成字符串去除多余0
            precision = len(tick_size_str.split('.')[1])  # 计算小数位数
        else:
            raise ValueError("无法获取交易对的精度信息")

        # 使用 precision 作为 round() 的第二个参数
        quantity = round(quantity * 0.97, precision)

        print(f"下单准备: symbol={symbol}, side={side}, quantity={quantity}, price={price}, stop_loss={stop_loss}, take_profit={take_profit}")

        # 1. 限价下单
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='LIMIT',
            timeInForce='GTC',  # GTC: 挂单直到成交或取消
            quantity=quantity,
            price=str(price)
        )
        print(f"限价下单成功: {order}")

        # 2. 止盈单
        take_profit_order = client.futures_create_order(
            symbol=symbol,
            side='SELL' if side == 'BUY' else 'BUY',
            type='TAKE_PROFIT_MARKET',
            stopPrice=str(take_profit),  # 触发止盈价格
            closePosition=True  # 全部平仓
        )
        print(f"止盈单设置成功: {take_profit_order}")

        # 3. 止损单
        stop_loss_order = client.futures_create_order(
            symbol=symbol,
            side='SELL' if side == 'BUY' else 'BUY',
            type='STOP_MARKET',
            stopPrice=str(stop_loss),  # 触发止损价格
            closePosition=True  # 全部平仓
        )
        print(f"止损单设置成功: {stop_loss_order}")

        print(f"限价单价格: {price}, 止盈价格: {take_profit}, 止损价格: {stop_loss}")
        return True

    except Exception as e:
        print(f"下单失败: {e}")
        return False