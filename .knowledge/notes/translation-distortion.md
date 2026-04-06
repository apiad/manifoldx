# Translation Distortion Analysis

## Symptoms
- Cube gets "squeezed" or "pyramid-like" when translated
- Bottom or right side vertices collapse

## Investigation So Far

### Python-side matrices are correct
Tested with various translations:
- Position correctly stored at matrix[3, 0:3]
- Scale remains uniform at 1.0
- Rotation applied correctly

### View-Projection Matrix
- Camera at default (0, 1, 2), target (0, 0, 0)
- FOV 60 degrees
- VP matrix correctly computed

### Test results:
```
Frame 0: pos=(0.000, 0.016, 0.000), scale=(1.000, 1.000, 1.000)
Frame 9: pos=(0.000, 0.160, 0.000), scale=(1.000, 1.000, 1.000)
```

### Possible Causes Not Ruled Out
1. Matrix upload to GPU - need to verify shader receives correct values
2. WGSL shader matrix multiplication
3. Vertex attribute interpretation
4. Depth buffer affecting vertex positions
5. Float precision on GPU

## Next Steps
- Check shader receives correct matrix values
- Test with simpler geometry (single triangle)
- Verify transpose is correct for WGSL
- Test without rotation, only translation
