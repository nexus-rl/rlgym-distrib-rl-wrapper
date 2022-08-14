from gym import register

id='RocketLeague-v0'
entry_point='rlgym_distrib_rl_wrapper.RLGymEnvironment:RLGymEnvironment'

register(id=id, entry_point=entry_point)
