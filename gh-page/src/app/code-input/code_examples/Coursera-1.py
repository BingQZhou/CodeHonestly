import simplegui

# formatting time function
def format(t):
    a = t // 600
    b = (t // 100) % 6
    c = (t // 10) % 10
    d = t % 10
    # string representation of the formatted time
    clock = str(a) + ':' + str(b) + str(c) + '.' + str(d)
    return clock

# event handlers for buttons
def start():
    timer.start()

def stop():

    if (timer.is_running() == True):	# controller for if stopwatch is started first
        if ( d == 0 ):
            x += 1
        y += 1
    result = str(x) + '/' + str(y)
    timer.stop()


def reset():
    count = 0
    format(count)
    x = 0
    y = 0
    result = str(x) + '/' + str(y)
    timer.stop()

# event handler for timer with 0.1 sec interval
timer = simplegui.create_timer(100, counter)

# draw handler
def draw(canvas):
    canvas.draw_text(format(count), [125, 125], 25, "White")
    canvas.draw_text(result, [225,50], 25, "Red")
