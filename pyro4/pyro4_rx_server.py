import Pyro4
import base64

import time

@Pyro4.expose
class server_callback(object):

	def response(self, data):

		print('Server Response : {}'.format(time.time()))

pyroDaemon = Pyro4.Daemon()

ns = Pyro4.locateNS()

uri = pyroDaemon.register(server_callback)

ns.register('Server', uri)

print('Server URI : {}'.format(uri))

pyroDaemon.requestLoop()
