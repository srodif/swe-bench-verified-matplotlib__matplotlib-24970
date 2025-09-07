"""
Test for NumPy 1.24 deprecation warning fix in colormap indexing.

This test specifically targets the fix for out-of-bound integer assignments
that cause deprecation warnings in NumPy 1.24+.
"""

import numpy as np
import pytest

import matplotlib.colors as mcolors


def test_uint8_colormap_no_warnings():
    """
    Test that applying a colormap to uint8 arrays doesn't generate
    NumPy 1.24+ deprecation warnings about out-of-bound integer conversion.
    
    This is a regression test for the GitHub issue about NumPy 1.24 
    deprecation warnings when using plt.get_cmap()(np.empty((0, ), dtype=np.uint8)).
    """
    # Create a simple colormap 
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'test', ['blue', 'red'], N=256)
    
    # Test with empty uint8 array (original issue)
    empty_uint8 = np.empty((0,), dtype=np.uint8)
    
    # This should not raise warnings in NumPy 1.24+
    with pytest.warns(None) as record:
        result = cmap(empty_uint8)
    
    # Filter out any unrelated warnings
    numpy_warnings = [w for w in record 
                     if 'NumPy will stop allowing conversion of out-of-bound' in str(w.message)]
    
    assert len(numpy_warnings) == 0, (
        f"NumPy deprecation warnings detected: {[str(w.message) for w in numpy_warnings]}")
    
    # Verify the result is correct shape (empty array with 4 columns for RGBA)
    assert result.shape == (0, 4)
    
    # Test with some actual uint8 values
    test_values = np.array([0, 127, 255], dtype=np.uint8)
    
    with pytest.warns(None) as record:
        result = cmap(test_values)
    
    numpy_warnings = [w for w in record 
                     if 'NumPy will stop allowing conversion of out-of-bound' in str(w.message)]
    
    assert len(numpy_warnings) == 0, (
        f"NumPy deprecation warnings detected: {[str(w.message) for w in numpy_warnings]}")
    
    # Verify the result is correct shape
    assert result.shape == (3, 4)
    

def test_colormap_dtype_preservation():
    """
    Test that the fix preserves the intended overflow behavior for different dtypes.
    """
    cmap = mcolors.LinearSegmentedColormap.from_list('test', ['blue', 'red'], N=256)
    
    # Test that the behavior is consistent regardless of input dtype
    test_cases = [
        np.array([255], dtype=np.uint8),
        np.array([255], dtype=np.uint16), 
        np.array([255], dtype=np.int32),
        np.array([255.0], dtype=np.float64),
    ]
    
    results = []
    for test_case in test_cases:
        with pytest.warns(None) as record:
            result = cmap(test_case / 255.0)  # Normalize to [0,1] range
        
        numpy_warnings = [w for w in record 
                         if 'NumPy will stop allowing conversion of out-of-bound' in str(w.message)]
        
        assert len(numpy_warnings) == 0, (
            f"NumPy deprecation warnings for dtype {test_case.dtype}: "
            f"{[str(w.message) for w in numpy_warnings]}")
        
        results.append(result)
    
    # All results should be similar (allowing for floating point differences)
    for i in range(1, len(results)):
        np.testing.assert_allclose(results[0], results[i], rtol=1e-10)