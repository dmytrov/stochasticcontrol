import time
import psychopy.visual as pv
import statemachine.machine as sm


class TextMessage(sm.State):
    def __init__(self, message="Hello, world!", timeout=1.0, nextstate=None):
        super(TextMessage, self).__init__()
        self.message = message
        self.timeout = timeout
        self.nextstate = nextstate
        self._pplabel = None
        self._enter_time = None

    def get_print_info(self):
        return self.message

    def on_enter(self):
        self._pplabel = pv.TextStim(win=self.statemachine.env.win, text=self.message, height=0.05) # units="pix")
        self._enter_time = time.time()

    def on_render(self):
        self._pplabel.draw()
        if time.time() > self._enter_time + self.timeout:
            self.statemachine.switch_state(self.nextstate)


class Sequence(sm.State):
    def __init__(self, states=None):
        super(Sequence, self).__init__()
        self.states = states  # list of states to iterate
        self.index = -1
        self.on_step = None  # callback
        self.on_sequence_complete = None  # callback
        
    def get_print_info(self):
        return "{}/{}".format(self.index+1, len(self.states))

    def on_enter(self):
        self.index += 1
        if self.index < len(self.states):
            if self.on_step is not None:
                self.on_step(self)
            self.statemachine.push_state(self.states[self.index])
        else:
            self.statemachine.pop_state()
            if self.on_sequence_complete is not None:
                self.on_sequence_complete(self)



