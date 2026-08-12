import tempfile
import numpy as np
from m2n import SymMatikaExplicitRequest, prepare_symmatika_job

x=np.linspace(-2,2,200)
X=np.column_stack([x, np.sin(x)])
y=x*x + 0.5*np.sin(x)
req=SymMatikaExplicitRequest(X,y,('x','sin_x'),'y',population_size=1000,max_depth=3)
with tempfile.TemporaryDirectory() as d:
    job=prepare_symmatika_job(req,d)
    print('dataset :',job.dataset_csv)
    print('driver  :',job.driver_cpp)
    print('contract:',job.contract_json)
    print('warning :',job.warning)
