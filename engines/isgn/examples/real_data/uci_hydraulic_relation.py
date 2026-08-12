import sys
from isgn.bridges.uci_hydraulic import load_uci_hydraulic
from isgn.algebraic import discover_algebraic_relation_subset

source=sys.argv[1]
train=load_uci_hydraulic(source,scope='cooling-power',split='train')
val=load_uci_hydraulic(source,scope='cooling-power',split='val')
result=discover_algebraic_relation_subset(train,val,inputs=['TS3','TS4','FS2'],target='CP',degree=2,max_terms=2)
print(result.to_dict())
