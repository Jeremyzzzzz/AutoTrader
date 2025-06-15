from binance.client import Client

# Binance API 密钥
api_key = 'QbqwCVv6p1cJEcvtAGhEfVPxaxR841JC7xyafG6kklLyCToIk05cYpVYz7zNcb9E'
api_secret = 'YzawJZuHmKj383VTBsRP1QkU19H9CkWUPw8Tqzng8UaVP6A9CRiszFeTerPkIejT'

# 初始化 Futures 客户端
client = Client(api_key, api_secret)

def get_client():
    """获取币安客户端实例"""
    return client