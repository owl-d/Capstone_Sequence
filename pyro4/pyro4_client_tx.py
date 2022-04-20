import Pyro4
import base64

import time

tx_node = Pyro4.Proxy('PYRONAME:Server')

tx_node.response('test')
