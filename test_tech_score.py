import sys, os
sys.path.insert(0, '/home/team/shared')
from thetaedge import compute_technical_score, TechnicalIndicators, OptionRecommendation

rec = OptionRecommendation(ticker='TEST', strike=100, current_price=110)
rec.technicals = TechnicalIndicators()
rec.technicals.trend = 'bullish'
rec.technicals.rsi = 45
rec.technicals.atr_pct = 5.0

score = compute_technical_score(rec, 'csp')
print(f'Bullish + RSI 45 + ATR 5% = {score:.0f}/100')

rec.technicals.trend = 'bearish'
rec.technicals.rsi = 25
score = compute_technical_score(rec, 'csp')
print(f'Bearish + RSI 25 + ATR 5% = {score:.0f}/100')

print('OK')