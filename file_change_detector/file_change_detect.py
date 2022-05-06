import os
import time

filename = input('File Directory : ')

print('Watching : {}'.format(filename))

last_modified_time = int(os.path.getmtime(filename))

while(True):

	if last_modified_time != int(os.path.getmtime(filename)):

		print('[File Changed]')

		break
