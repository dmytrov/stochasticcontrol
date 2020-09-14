"""
Simple state machine to run experiments
"""

class State(object):
    def __init__(self):
        super(State, self).__init__()
        self.statemachine = None

    def get_print_info(self):
        return ""

    def _set_statemachine(self, statemachine):
        self.statemachine = statemachine
        self.on_init()

    def on_init(self):
        """
        Called only once
        """
        pass

    def _on_enter(self):
        if self.statemachine.env.logger is not None:
            self.statemachine.env.logger.info("State:" + self.__class__.__name__ + ".on_enter")
        self.on_enter()
        
    def on_enter(self):
        """
        Called on every state re-entring
        """
        pass

    def _on_leave(self):
        if self.statemachine.env.logger is not None:
            self.statemachine.env.logger.info("State:" + self.__class__.__name__ + ".on_leave")
        self.on_leave()

    def on_leave(self):
        """
        Called on every state leave
        """
        pass

    def on_render(self):
        pass



class StateMachine(object):
    def __init__(self, env):
        """
        env: container for shared objects the states need access to
        """
        super(StateMachine, self).__init__()
        self.env = env
        self.statestack = []
        
    def _print_statestack(self):
        print(["{}({})".format(state.__class__.__name__, state.get_print_info()) for state in self.statestack])

    def get_current_state(self):
        if len(self.statestack) > 0:
            return self.statestack[-1]
        else:
            return None

    def push_state(self, newstate):   
        cs = self.get_current_state()
        if cs is not None:
            cs._on_leave()
        self.statestack.append(newstate)
        if self.env.is_debug:
            self._print_statestack()
        newstate._set_statemachine(self)
        newstate._on_enter()


    def pop_state(self):
        cs = self.get_current_state()
        if cs is not None:
            cs._on_leave()
            self.statestack.pop()
            if self.env.is_debug:
                self._print_statestack()
        cs = self.get_current_state()
        if cs is not None:
            cs._on_enter()
            
        
    def switch_state(self, newstate):
        cs = self.get_current_state()
        if cs is not None:
            cs._on_leave()
            self.statestack.pop()
            if self.env.is_debug:
                self._print_statestack()
        if newstate is not None:
            self.statestack.append(newstate)
            if self.env.is_debug:
                self._print_statestack()
            newstate._set_statemachine(self)
            newstate._on_enter()            
        else:
            cs = self.get_current_state()
            if cs is not None:
                cs._on_enter()
                

    def _render_current(self):
        cs = self.get_current_state()
        if cs is not None:
            cs.on_render()

    def _render_all(self):
        for state in self.statestack:
            state.on_render()

    def render(self):
        self._render_all()
        #self._render_current()
    