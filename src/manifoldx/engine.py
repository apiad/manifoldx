class Engine:
    def __init__(self, name: str, h: int = 600, w: int = 800, fullscreen: bool = False):
        self.name = name
        self.h = h
        self.w = w
        self.fullscreen = fullscreen
        self._running = False
        self._startup_callbacks = []
        self._shutdown_callbacks = []

    def startup(self, func):
        self._startup_callbacks.append(func)
        return func

    def shutdown(self, func):
        self._shutdown_callbacks.append(func)
        return func

    def quit(self):
        self._running = False

    def run(self):
        self._running = True
        for callback in self._startup_callbacks:
            callback()
        while self._running:
            pass
        for callback in self._shutdown_callbacks:
            callback()