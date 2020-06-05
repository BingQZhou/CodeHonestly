# STOPWATCH: Time as you've never seen it before
# (I used to work in PR)

# requires SimpleGUICS2Pygame module
import simplegui
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

# define global variables

# define helper function
def format(t):
    A = time / 600
    B = ((time / 10) % 60) / 10
    C = ((time / 10) % 60) % 10
    D = time % 10
    t = str(A) + ":" + str(B) + str(C) + "." + str(D)
    return t

# define event handlers for buttons

def start_timer():
    timer.start()
    running = True

def stop_timer():
    timer.stop()
    if running == True:
        counter += 1
        if time % 10 == 0:
            success +=1
    running = False

def reset_timer():
    timer.stop()
    time = 0
    running = False
    counter = 0
    success = 0

# define event handler for timer with 0.1 sec interval

def timer_handler():
    time += 1
    print(time)

# define draw handler

def draw(canvas):
    t = "0:00.0"
    t = format(t)
    canvas.draw_text(t,[130,150],24,"Yellow","monospace")
    canvas.draw_text(str(success)+"/"+str(counter),[260,30],20,"Red","monospace")
