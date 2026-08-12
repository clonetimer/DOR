from isgn.simulation import simulate
from isgn.api import complete_graph

train,_=simulate('motor',20,T=241,dt=.03,seed=10)
val,_=simulate('motor',8,T=241,dt=.03,seed=20)
result=complete_graph(
    train,val,
    known_present={'i':['V'],'w':['TL']},
    known_absent={'i':['TL'],'w':['V']},
    backend='reference-sindy',
    backend_options={'normalize_columns':True},
)
print(result.summary.to_dict())
