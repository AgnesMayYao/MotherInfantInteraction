
from sklearn.metrics import cohen_kappa_score

annotation = []
prediction = []

'''
annotation.extend(['asleep']*(3284+446+133+24))
prediction.extend(['asleep']*3284)
prediction.extend(['off']*446)
prediction.extend(['on']*133)
prediction.extend(['dys']*24)

annotation.extend(['off']*(449+1275+647+198))
prediction.extend(['asleep']*449)
prediction.extend(['off']*1275)
prediction.extend(['on']*647)
prediction.extend(['dys']*198)

annotation.extend(['on']*(311+1600+2865+715))
prediction.extend(['asleep']*311)
prediction.extend(['off']*1600)
prediction.extend(['on']*2865)
prediction.extend(['dys']*715)

annotation.extend(['dys']*(66+319+285+412))
prediction.extend(['asleep']*66)
prediction.extend(['off']*319)
prediction.extend(['on']*285)
prediction.extend(['dys']*412)


annotation.extend(['walk']*(1524 + 7 + 148 + 2 + 2))
prediction.extend(['walk']*1524)
prediction.extend(['jog']*7)
prediction.extend(['stairs']*148)
prediction.extend(['sit']*2)
prediction.extend(['stand']*2)

annotation.extend(['jog']*(10 + 1280 + 31))
prediction.extend(['walk']*10)
prediction.extend(['jog']*1280)
prediction.extend(['stairs']*31)
prediction.extend(['sit']*0)
prediction.extend(['stand']*0)


annotation.extend(['stairs']*(185 + 33 + 784 + 4 + 4))
prediction.extend(['walk']*185)
prediction.extend(['jog']*33)
prediction.extend(['stairs']*784)
prediction.extend(['sit']*4)
prediction.extend(['stand']*4)


annotation.extend(['sit']*(10+272))
prediction.extend(['walk']*4)
prediction.extend(['jog']*0)
prediction.extend(['stairs']*2)
prediction.extend(['sit']*272)
prediction.extend(['stand']*4)


annotation.extend(['stand']*(3 + 1+ 10+ 0 + 209))
prediction.extend(['walk']*3)
prediction.extend(['jog']*1)
prediction.extend(['stairs']*10)
prediction.extend(['sit']*0)
prediction.extend(['stand']*209)
'''


print cohen_kappa_score(annotation, prediction)
