Today I want to tell you a different kind of story. It's not about machine learning, large language models, algorithms, or theory of computer science.

It's about a side project that I've been building for a couple of weeks that made me fall in love again with an ancient love of mine. The quick and easy way to explain it is this: a performance-focused graphics engine for data-driven visualizations in Python.

But that's only the surface. If you want to see the coold demos and the technical description, feel free to scroll down. But if you want to know the story behind it, let me start from the beginning.

## The Origin Story

So, this starts back in undergrad, before I did anything related to machine learning or optimization or statistics. My first love was actually computer graphics. I had "learned to code" like, I don't know, at 11 or 12, and for the first five years or so, before getting to college and actually learning to code, all my "coding" was basically tiny games. It was RPG Maker back then---who remembers that?

I always wanted to be a game developer, as you may imagine, and I think that's probably the main motivation why I studied Computer Science. There are two kind of people who want to study Computer Science, as a matter of fact. One is people who love games---and the other is, of course, people who hate games; there are no in-betweens.

I was the loving-games kind and all I wanted to build games for a life. So when I was in first year, after actually learning some real coding, my first kind of large project was a game engine. This was before Unity, before even XNA---who remembers that?---this was when .NET was getting started, and I wrote a quick and dirty game engine in pure C# that talked native DirectX 11.

It was very cheap, a disaster of architecture almost surely, but it taught me the basics of how to construct a scene graph, how to animate a camera, how to do lighting, how to write very basic shaders. I learned a ton and basically fell in love with computer graphics.

I ended up doing my diploma thesis in computer graphics---screen-based global illumination, a couple of years before NVIDIA came up with ray tracing on the GPU, which basically killed that whole area of research. And I also did my Master's on global illumination and some data structures for the GPU, but after graduation I quickly switched research towards machine learning and AI, which, you can imagine, this 2014, and deep learning was just on the rise. The rest is history, as they say.

And then, here I was this past week thinking about old projects that I used to have fun with when I was in college, and trying to remember what it felt like to code back then, no LLMs, no internet for the most part even. The time where I've been the most fun was probably when I dabbed into procedural generation of cities, mountains, lakes, and... stuff, in the late 2012. This was at the early era of PCG, and I never got to do anything with that other than a few tutorials and a few lessons that I taught at University.

I played with Unity for a couple of years, but nothing too serious--I think I was actually one of the first people in my University to even install Unity, and I even taught a couple of Master's courses on it. I participated in a couple of game jams, but after 2017 or so I stopped doing graphics all together. And I've been doing machine learning since.

But, in any case, I kind of forgot about computer graphics all along. At least during day-worked. So there was I last weeek, remembering that and asking how hard would it be to actually make a graphics engine in Python, some quick hack like my undergrad projects. I did a bit of research and I discovered that Python is, as of 2026, in a very good position to build a graphics engine, and not just a crapy one, but one that is actually fast. We have WGPU now---the spiritual and practical succesor of OpenGL (who remembers that?), which has native suppotr for GPU-accelerated graphics in Linux.

I basically did a plan and sat for three days to hack this thing.

## The Engine

My first idea was to have a Rust backend for all the graphics engine stuff---the rendering loop, materials, lights---but I quickly decided to drop that idea because getting Rust and Python to talk to each other was becoming increasingly harder and harder, and I really wanted to finally see a damn cube rendering on my screen.

So I decided to switch completely to Python. But since I'm a grown-up now, I have to find some kind of serious objective for making something like this. I decided I didn't want to make a typical graphics engine where you have a scene graph with hierarchies of entities and properties, and you simply render all of them. No, that is way too 2000s.

I decided I wanted to do a very fast, data-driven visualization tool purely based on the Entity-Component-System (ECS) paradigm and make it extremely performant, so it would focus on big data-driven simulations like N-body simulations, chemical and physics experiments, AI pathfinding and agents, you know, grown-up stuff like that.

(But actually, all I wanted was to play with WGPU and draw some cubes in Python. Wink, wink.)

This framing gave me two things, though. My solution doesn't have to be very fancy as a game engine, we don't need to be able to like load skeletal animations or stuff like that. It's not actually a game engine; it's a graphics engine with at best some interaction logic. But it still lets you do some cool stuff, even if all you can render is blocks and spheres. When you can render thousands of them running very fast on the GPU, you can do some cool stuff. So this is the motivation, and now let me show you what I have.

### Deep Dive

So here is **manifold**---short Manifold Graphics if you want. It's a Python library built on top of WGPU, a graphics engine based on the Entity-Component-System paradigm.

If you have never heard about it, ECS is a completely different way of writing code that is especially tailored for video games, but it is very little known outside of the game development world. And its awesome.

In a typical business code, you have entities who own their data, and you usually have behavior associated to entities; so entities also own their behavior---this is the basic Object-Oriented Programming paradigm where objects own their data and their methods. And if you want to do something with an object, you have to call methods on the object so the object guarantees the instance invariances.

Since OOP was basically the ONE programming paradigm of the 90s and early 2000s---when the videogame industry really exploded---it is only normal that we started writing games like this. But there is a problem with OOP (well, many problems, but one in particular that matters for our discussion).

When you have 10,000 objects, each of them with more or less the same structure, e.g., they are physical particles bouncing with each other, or little zerlings comming to your base, you simply _cannot_ update them fast enough. For example, making a physics simulation out of this is extremely slow if you have to go to each particle and update its velocity, its scale, its rotation, etc. **You**'ll end up doing thousands of tiny method calls, thrashing your cache, and issuing lots of super small copies to GPU for drawing.

What you want is to vectorize this operation. You would like to have all of the objects' data in a single NumPy matrix, and you want to write a very, very efficient vectorized code that doesn't do any loop and just updates everything at once. THen copy all the data to the GPU and issue a single draw call that renders all objects parameterized by their positions, rotations, etc. Chef kiss.

This is the Entity-Component-System paradigm at its core. It completely flips the responsibilities from standard OOP The **components** are just flat storage of data (rows in a matrix) and the **entities** are just pointers to a row where all of their data lives. Then the systems are methods that act on a subset of entities using heavily vectorized code, because each system deals with a large number of equally-structured entities, and they don't care which is which.

Im **manifoldx**, each system is a Python method that receives a subset of entities that have some combination of components. For example, if you want to process all of the particles in a simulation, you write a system that receives entities that have the `Particle` component, perhaps also a `Transform` component. In the transform component, you will have the position, rotation, scale, and the particle component will store simulation-specific data like velocity, temperature, momentum, etc.

The key to high performance in ECS is to avoid looping as much as possible. You assume all of the components of the entities in a system have exactly the same layout, so what you get is really a view of a matrix, and you write vectorized code. You add something to all them, you multiply all them by something, or in general you compute some matrix operations on them. All at once.

And if you can write your code like this, then you get a very, very fast rendering loop because instead of making one method invocation per entity, you make one method invocation per _archetype_, that is, per combination of components, which is a couple of order of magnitude less that your entities count.

Here's a minimal example showing how the ECS works in **manifoldx**:

<!-- Use spheres.py here is an example instead of these fake cubes -->

```python
import manifoldx as mx
import numpy as np

engine = mx.Engine("Cubes")

# Define a custom component
@engine.component
class Particle:
    velocity: mx.Vector3


N = 1000

# Create lots of entities
engine.spawn(
    mx.geometry.cube(), # with the same mesh
    mx.material.phong(mx.colors.RED), # the same material
    # and random data
    Transform(pos=np.random.uniform(-1,1,size-(N, 3))),
    Particle(velocity=np.random.uniform(-1,1,size=(N,3))),
    n=N,  # spawn 1000 at once with instanced rendering
)

# Systems are methods that receive a query of entities
@engine.system
def move_things(query: mx.Query[Transform, Particle], dt: float):
    # This is a SINGLE vectorized operation over all N entities
    query[Transform].pos += query[Particle].velocity * dt

engine.run()
```

That's it. A single line of code to update all positions at once. Notice the `query` argument that defines which entities you get (all entities with both a `Transform` and a `Particle` component).

In a real simulation, you can have, say 10 systems, but you have 10,000 or 100,000 entities, and you know you can do very fast vetorized updates in NumPy for all them, 10 times each frame.

For example, if you have 500 particles and you want to do N-body simulation, computing the 500-squared gravity interactions 60 times per second in Python is suicide. But if you do it in NumPy, then you get something that runs in a few milliseconds. A quarter million interactions computed 60 times per second. In Python.

To make it really efficient, you need to also avoid copying or moving data; it's all masking and clever NumPy layout that keeps all of the memory in one place, and you are just seeing fragments of that memory in each system.

The other key idea is that you don't modify anything in a system. That line where position is set, doesn't really write back to the matrix. All it does is compute the right-hand side and then you issue a command that will be run at the end of all the systems, before frame rendering happens. This allows to write pure threaded parallelism, because you can run several systems in different threads---they are all reading the same data, but they aren't writing to the buffers, which is great since Python has real support for multi-threading now in 2026 (after 35 years!).

## Showcase

That is the basic idea. Now lets see some examples. AS of today, version 0.2, **manifoldx** has some basic shapes like cubes, spheres, and planes, and support for basic PBR lighting, camera controls, and that's basically it.

All the engine realy does is set up this somewhat clever inversion of logic that forces you to write very efficient code, and the magic is in what you do inside the systems.

So let me show you three examples.

### 1. N-Body Gravitational Simulation

The first is an N-body simulation. All gravity computation happens in a single NumPy block with no Python loops. The only relevant part of the code is the gravity system, that looks something like this.

```python
@engine.system
def nbody_gravity(query, dt):
    pos = query[Transform].pos.data  # (N, 3)

    # All-pairs position differences: (N, N, 3)
    diff = pos[None, :] - pos[:, None]
    dist = np.linalg.norm(diff, axis=2)

    # Force magnitude: G * m_i * m_j / r²
    force_mag = G * mass_prod / np.maximum(dist, SOFTENING)**2

    # Net force = sum over all other bodies
    net_force = (force_mag[:, :, None] * diff / dist[:, :, None]).sum(axis=1)

    velocities += (net_force / masses[:, None]) * dt
    query[Transform].pos += velocities * dt
```

This runs 500 bodies with 250,000 force pair computations at 60fps.

### 2. Ideal Gas Simulation

The second example is an ideal gas with elastic collisions inside a bounding box. Again, all running without a single for loop. Collision detection and impact resolution in vectorizednumpy operations.

```python
@engine.system
def gas_physics(query, dt):
    pos = query[Transform].pos.data

    # Wall collisions: vectorized mask
    below = (pos + velocities * dt) < -BOX_HALF
    above = (pos + velocities * dt) > BOX_HALF

    # Here we avoid branching and use masking instead
    velocities[below] = np.abs(velocities[below]) * RESTITUTION
    velocities[above] = -np.abs(velocities[above]) * RESTITUTION

    # Particle collisions: find overlapping pairs
    diff = pos[None, :] - pos[:, None]
    dist = np.linalg.norm(diff, axis=2)
    overlap = dist < 2 * PARTICLE_RADIUS
    i_idx, j_idx = np.where(np.triu(overlap))

    # Resolve collisions with impulse
    # ... (collision resolution code)
    # ... (also vectorized)

    query[Transform].pos += velocities * dt
```

### 3. Boids Flocking

The third example is a Boids simulation with emergent flocking behavior. This is the one that strikes me the most because boids simulation is often compute-heavy. Each individual entity must keep track of a subset of neighbors and adjust behavior based on them, not the whole set of entities. But again, a bit of numpy magic lets us vectorize the crap out of this and simulate 300 boids at 60 frames per second.

```python
@engine.system
def boids_physics(query, dt):
    # Separation, alignment, cohesion as vectorized tensor ops
    diff = pos[None, :] - pos[:, None]  # (N, N, 3)
    dist_sq = (diff * diff).sum(axis=2)

    neighbors = dist_sq < PERCEPTION_SQ

    # Separation (1/dist² weighted)
    sep = (-diff * (neighbors[:,:,None] * inv_dsq[:,:,None])).sum(axis=1)

    # Alignment (average neighbor velocity)
    avg_vel = (vel[None,:] * neighbors[:,:,None]).sum(axis=1) / safe_count

    # Cohesion (steer toward center of mass)
    center = (pos[None,:] * neighbors[:,:,None]).sum(axis=1) / safe_count

    # Plus predator avoidance and boundary steering...
    # That one is easy.
```

You can check all the examples in the [Github](https://github.com/apiad/manifoldx) repository to see the full code, but the bulk of the implementation is these cleverly vectorized system methods.

## Future Directions

And that's it for today. This is my pure Python (well, you know what I mean) graphics engine for serious, grown-up stuff that is surely not a weekend side-project.

Where I will go with this? I don't know. I always write these things mostly as a learning exercise and I've learned a lot about graphics in Python. I've updated my view of modern graphics and I think I've paid my debt of the last seven years that I haven't done any graphics computation. I'm kind of happy now that I know how to do this.

There are some places this engine can go to, like rendering custom shaders when you need stuff like lighting effects. But it is not going to become a traditional, full-blown game engine. I will not add support for lots of game engine-like features including, I don't know, skeletal animations, level of detail, scene management, or, god forbids, visual scripting or any nonsense like that.

There are two areas which I do believe I would like to explore in the future. One is extending the engine towards the kind of behavior you need to write for AI simulations, like if you want to run some sort of agent simulation or ant colony optimization or stuff like that. That code doesn't look that much as a frame-by-frame update, but it looks more like an asynchronous event-kind of code, which is also something that is not usual in game engines. You cannot do async await for some event to happen. That is one direction to go.

And the other direction to go is towards procedural generation of meshes and content in general, which is an area I left five or six years ago—haven't come back to it. And there is so much that I want to do in the sense of, I don't know, building infinite worlds with infinite trees that keep expanding on-site, stuff like that, just for the sake of fun.

And that's it for this week. This is not production-ready at all—it's mostly a toy at the moment—but you can take it apart and hack your way into some cool physics or mathematical simulation. The code is on GitHub if you want to try it yourself, and I'd love to hear what you build with it.

Until next week, stay curious.
