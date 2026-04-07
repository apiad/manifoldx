import manifoldx as mx

engine = mx.Engine("Hello World", height=600, width=800, fullscreen=False)

@engine.startup
def init():
    print("Hello World")
    engine.quit()

@engine.shutdown
def close():
    print("Shutting down")

engine.run()
