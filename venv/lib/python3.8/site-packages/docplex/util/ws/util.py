# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2019-2020
# --------------------------------------------------------------------------

# gendoc: ignore

import json
import uuid
try:
    from IPython.display import Javascript, HTML, display
except ImportError:
    Javascript = None

"""
There are two versions of the segments informations.
Version 1 in the version used until May 2020
Version 2 is the version used after May 2020.

During the transition time, we will generate segments tracking for both versions.
After every infrastructure is in version 2, we will remove version 1.

"""
    
# Those were the initial values
CATEGORY = "Decision Optimization"
START_SOLVE_EVENT = "Solve in Notebook"
END_SOLVE_EVENT = "End solve in Notebook"
# Process values for version-2
PROCESSES = {START_SOLVE_EVENT: 'Started Process',
             END_SOLVE_EVENT: 'Ended Process'}

def generate_payload(category, event, details=None, hw_spec=None, version=1):
    if version==1:
        payload = {'action' : event,
                   'model': {'type': 'python'},
                   'category': category }
        if details:
            payload['details'] = details
        if hw_spec:
            payload['hardware_spec'] = hw_spec
    elif version==2:
        payload = {
            'platformTitle': "Watson Studio",
            'productTitle': "Decision Optimization",
            'processType': "do_solve",
            'process': "do_solve/{modelType}/{solveId}".format(**details),
            'successFlag': True, # success in starting the process
            'custom.modelType': "python",
            'custom.solveId': details['solveId']
        }
        if event == START_SOLVE_EVENT:
            payload.update({'custom.modelSize.numConstraints': details['modelSize']['numConstraints'],
                            'custom.modelSize.numVariables': details['modelSize']['numVariables'],
                            'custom.engineEdition': details['edition'],
                            'custom.engineType': details['modelType'],})
        elif event == END_SOLVE_EVENT:
            payload.update({'custom.solveTime': details['solveTime']})
    else:
        raise ValueError("'version' is not in valid range")
    return json.dumps(payload, indent=3)

def generate_js(category, event, details=None, version=1):
    '''
    For versions:
    
       - 1: refers to the old action type
       - 2: refers to the new action type see https://github.ibm.com/IBMDecisionOptimization/dd-planning/issues/2266
    
    Args:
        version: 1 if older segment type, 2 if new segment type.
    '''
    if version == 1:
        template ='''
            if (parent && parent.analytics)
                parent.analytics.track("{category}: {event}",
                                        {payload})
        '''.format(payload=generate_payload(category, event, details),
                   category=category, event=event)
    elif version == 2:
        template ='''
            if (parent && parent.analytics)
                parent.analytics.track("{process}",
                                        {payload})
        '''.format(payload=generate_payload(category, event, details, version=2),
                   process=PROCESSES[event])

    else:
        raise ValueError("'version' is not in valid range")
    return template


class Tracker(object):    
    # Amplitude tracker to get instrumentation when running on WS
    def notify_ws(self, event, details=None):
        if Javascript:
            js = generate_js(CATEGORY, event, details)
            uid = str(uuid.uuid4())
            display(Javascript(js), display_id=uid)
            display(HTML('<div></div>'), update=True, display_id=uid)
            # send also new segment types
            js = generate_js(CATEGORY, event, details, version=2)
            uid = str(uuid.uuid4())
            display(Javascript(js), display_id=uid)
            display(HTML('<div></div>'), update=True, display_id=uid)
