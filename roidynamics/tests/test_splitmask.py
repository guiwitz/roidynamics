from pathlib import Path
from microfilm.dataset import MultipageTIFF
import skimage
from roidynamics import splitmask
import numpy as np

image_path = Path('roidynamics/tests/test_folders/test_multipage_good/')
roi_path = Path('roidynamics/tests/test_folders/test_multipage_good/roi.bmp')
multi_roi_path = Path('roidynamics/tests/test_folders/test_multipage_good/multiroi.bmp')
concentric_path = Path('roidynamics/tests/test_folders/concentric_mask_w4_n3.tif')
concentric_path2 = Path('roidynamics/tests/test_folders/concentric_mask_w4_n3_b.tif')


image = MultipageTIFF(image_path)
roi = skimage.io.imread(roi_path)

def test_get_roi_cm():
    cm = splitmask.get_roi_cm(roi_im=roi)
    np.testing.assert_array_equal(cm, np.array([[70,90]]))

def test_get_roi_cm_path():
    cm = splitmask.get_roi_cm(roi_path=roi_path)
    np.testing.assert_array_equal(cm, np.array([[70,90]])) 

def test_save_labelled_roi():
    splitmask.save_labelled_roi('myfile.tif', roi_path=multi_roi_path)
    im = skimage.io.imread('myfile.tif')
    assert im.shape == (196, 171)
    assert im.max() == 4

def test_create_concentric_mask():
    concentric_mask_test = splitmask.create_concentric_mask(
        center=[30,40], im_dims=[100,120], sector_width=4, num_sectors=3)
    concentric_mask = skimage.io.imread(concentric_path)
    np.testing.assert_array_equal(concentric_mask_test, concentric_mask)

def test_create_concentric_mask2():
    concentric_mask_test = splitmask.create_concentric_mask(
        center=[[30,40], [60,70]], im_dims=[100,120], sector_width=4,
        num_sectors=3)
    concentric_mask = skimage.io.imread(concentric_path)
    concentric_mask2 = skimage.io.imread(concentric_path2)
    np.testing.assert_array_equal(concentric_mask_test[0], concentric_mask[0])
    np.testing.assert_array_equal(concentric_mask_test[1], concentric_mask2[0])

def test_create_sector_mask():
    sector_mask = splitmask.create_sector_mask(center=[30,40], im_dims=[100,120], angular_width=20, max_rad=20, ring_width=None)
    assert sector_mask.max() == 18, f"expected 18 sectors, got {sector_mask.max()}"

def test_create_sector_mask_ring():
    sector_mask = splitmask.create_sector_mask(center=[30,40], im_dims=[100,120], angular_width=20, max_rad=20, ring_width=10)
    assert np.sum(sector_mask[0][29,:]== 18) == 10, "Wrong ring width or value"

def test_measure_intensities():
    time_concentric = skimage.io.imread('splitmask/tests/test_folders/concentric_time.tif')
    concentric_label = splitmask.create_concentric_mask(
        center=[50,50], im_dims=[120,100], sector_width=2, num_sectors=10)
    measure_int = splitmask.measure_intensities(time_image=time_concentric[np.newaxis, :,:,:], im_labels=concentric_label,
            channels=['0'], min_time=0, max_time=None, step=1)

    assert measure_int.shape == (20,10,1,1), "Wrong intensity output shape"
    assert np.ravel(measure_int[0])[1] == 100, "Wrong intensity value in window with 100 percent signal"
    assert np.ravel(measure_int[0])[3] == 0, "Wrong intensity value in window with 0 percent signal"