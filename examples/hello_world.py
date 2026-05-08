import manifoldx as mx

engine = mx.Engine("Hello World", height=600, width=800, fullscreen=False)

@engine.on("startup")
def init(_payload):
    print("Hello World")
    engine.quit()

@engine.on("shutdown")
def close(_payload):
    print("Shutting down")

engine.run()
