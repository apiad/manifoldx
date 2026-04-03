import glfw


class Engine:
    def __init__(self, name: str, h: int = 600, w: int = 800, fullscreen: bool = False):
        self.name = name
        self.h = h
        self.w = w
        self.fullscreen = fullscreen
        self._running = False
        self._window = None
        self._startup_callbacks = []
        self._shutdown_callbacks = []
        self._update_callbacks = []

    def startup(self, func):
        self._startup_callbacks.append(func)
        return func

    def shutdown(self, func):
        self._shutdown_callbacks.append(func)
        return func

    def update(self, func):
        self._update_callbacks.append(func)
        return func

    def quit(self):
        self._running = False

    def run(self):
        glfw.init()
        self._window = glfw.create_window(self.w, self.h, self.name, None, None)
        glfw.make_context_current(self._window)

        self._running = True
        for callback in self._startup_callbacks:
            callback()
        while self._running:
            pass
        for callback in self._shutdown_callbacks:
            callback()

        if self._window:
            glfw.destroy_window(self._window)
        glfw.terminate()