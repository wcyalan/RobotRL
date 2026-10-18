"""Predeclared commands for behavior acceptance (time in seconds)."""
def target_command(profile, time_s):
    if profile == 'turn_left':
        return (0., 0., 0.5)
    if profile == 'turn_right':
        return (0., 0., -0.5)
    if profile == 'stop_restart':
        return (0.5 if int(time_s/5) % 2 == 0 else 0., 0., 0.)
    raise ValueError(profile)
