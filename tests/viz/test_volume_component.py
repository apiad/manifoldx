"""Volume component layout: one int-valued handle per entity."""
import numpy as np

from manifoldx.viz import Volume


def test_volume_layout_is_single_int_field():
    """Volume holds exactly one field 'volume_id' at offset 0, length 1."""
    assert Volume._layout == {"volume_id": (0, 1)}


def test_volume_construct_with_handle():
    """Volume(volume_id=7) materializes a single-entity component
    with the value 7 in the volume_id slot.
    """
    v = Volume(volume_id=7)
    data = v.get_data(1)
    assert data.shape == (1, 1)
    assert int(data[0, 0]) == 7


def test_volume_construct_with_array():
    """Volume(volume_id=array_of_N) materializes N entities."""
    handles = np.array([1, 2, 3], dtype=np.int32)
    v = Volume(volume_id=handles)
    data = v.get_data(3)
    assert data.shape == (3, 1)
    assert (data[:, 0].astype(np.int32) == handles).all()
