# Sun / Earth / Moon Assets

Equirectangular albedo textures for `examples/sun_earth_moon_demo.py`.

## earth.jpg

- 2048 × 1024
- Source: NASA Visible Earth, "Shallow Topography" (land + shallow ocean), https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_2048.jpg
- License: NASA imagery is in the public domain.

## moon.jpg

- 1024 × 512
- Source: NASA GSFC Scientific Visualization Studio, LRO color shaded relief, https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_poles_1k.jpg
- License: NASA imagery is in the public domain.

## Sun

The sun in the demo is an unlit yellow `BasicMaterial` — `BasicMaterial`
doesn't sample textures yet (that's follow-up work in a future plan).
A point light at the sun's center handles its lighting role for Earth
and Moon.
