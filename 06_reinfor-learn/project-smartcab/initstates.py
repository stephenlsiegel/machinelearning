def init_states():
	states = {}
	waypoints = ['forward', 'left', 'right']
	trafficlight = ['red','green']
	oncoming = [None, 'left', 'right', 'forward']
	left = [None, 'left', 'right', 'forward']

	# Create dictionary of all possible states
	for w in waypoints:
		for t in trafficlight:
			for o in oncoming:
				for l in left:
					new_key = (w,t,o,l)
					#if w=='forward':
					#    self.states[new_key] = {None:0.5, 'forward':0.5, 'right':0, 'left':0}
					#elif w=='right':
					#    self.states[new_key] = {None:0.5, 'forward':0, 'right':0.5, 'left':0}
					#elif w=='left':
					#    self.states[new_key] = {None:0.5, 'forward':0, 'right':0, 'left':0.5}
					states[new_key] = {None:0.0, 'forward':0.0, 'right':0.0, 'left':0.0}
	
	return states