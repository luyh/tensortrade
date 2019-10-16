STRAGETY = 'tf'  # 'sb'

if STRAGETY == 'tf':    from ..strategy.tf_strategy import strategy as strategy
elif STRAGETY == 'sb':  from ..strategy.sb_strategy import strategy as sb_stragety
else: raise NotImplemented

performance = strategy.run(episodes=2,
                           testing=True)

print(performance[-5:])
print('done')