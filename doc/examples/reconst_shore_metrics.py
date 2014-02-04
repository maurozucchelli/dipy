"""
===========================
Calculate SHORE scalar maps
===========================

We show how to calculate two SHORE-based scalar maps: return to origin
probability (rtop) [Descoteaux2011]_, return to axis probability (rtap)
[Ozarslan2013]_, return to plane probability (rtpp) [Ozarslan2013]_,
propagator anisotropy (pa) [Ozarslan2013]_, propagator non-gaussianity (png)
[Ozarslan2013]_ and mean square displacement (msd) [Wu2007]_, [Wu2008]_
on your data. SHORE can be used with any multiple b-value dataset like
multi-shell or DSI.

First import the necessary modules:
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi, get_sphere
from dipy.data import get_data, dsi_voxels
from dipy.reconst.shore import ShoreModel

"""
Download and read the data for this tutorial.
"""

fetch_taiwan_ntu_dsi()
img, gtab = read_taiwan_ntu_dsi()

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values). For example, to read the b-values
it is possible to write print(gtab.bvals).

Load the raw diffusion data and the affine.
"""

data = img.get_data()
affine = img.get_affine()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
Instantiate the Model.
"""

asm = ShoreModel(gtab)

"""
Lets just use only one slice only from the data.
"""

dataslice = data[30:70, 20:80, data.shape[2] / 2]

"""
Fit the signal with the model and calculate the SHORE coefficients.
"""

asmfit = asm.fit(dataslice)

"""
Calculate the analytical rtop on the signal
that corresponds to the integral of the signal.
"""

print('Calculating... rtop_signal')
rtop_signal = asmfit.rtop_signal()

"""
Now we calculate the analytical rtop on the propagator,
that corresponds to its central value.
"""

print('Calculating... rtop_pdf')
rtop_pdf = asmfit.rtop_pdf()

"""
In theory, these two measures must be equal,
to show that we calculate the mean square error on this two measures.
"""

mse = np.sum((rtop_signal - rtop_pdf) ** 2) / rtop_signal.size
print("mse = %f" % mse)

"""
mse = 0.000000

Is possible to calculate also the rtap an the rtpp in the same way.
First the reconstruction sphere is needed.
"""

sphere = get_sphere('symmetric724')
print('Calculating... rtap')
rtap = asmfit.rtap(sphere)
print('Calculating... rtpp')
rtpp = asmfit.rtpp(sphere)


"""
Show the first maps and save them in SHORE_maps_1.png.
"""

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(2, 2, 1, title='rtop_signal')
ax1.set_axis_off()
ind = ax1.imshow(rtop_signal.T, interpolation='nearest', origin='lower')
ax2 = fig.add_subplot(2, 2, 2, title='rtop_pdf')
ax2.set_axis_off()
ind = ax2.imshow(rtop_pdf.T, interpolation='nearest', origin='lower')
ax3 = fig.add_subplot(2, 2, 3, title='rtap')
ax3.set_axis_off()
ind = ax3.imshow(rtap.T, interpolation='nearest', origin='lower')
ax4 = fig.add_subplot(2, 2, 4, title='rtpp')
ax4.set_axis_off()
ind = ax4.imshow(rtpp.T, interpolation='nearest', origin='lower')
plt.savefig('SHORE_maps_1.png')

"""
Let's calculate the analytical mean square displacement, the propagator anisotropy
and the non-gaussianity index.
"""
print('Calculating... pa')
pa = asmfit.propagator_anisotropy()
print('Calculating... png')
png = asmfit.propagator_non_gaussianity()
print('Calculating... msd')
msd = asmfit.msd()


"""
Show these maps and save them in SHORE_maps_2.png.
"""

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(2, 2, 1, title='pa')
ax1.set_axis_off()
ind = ax1.imshow(pa.T, interpolation='nearest', origin='lower')
ax2 = fig.add_subplot(2, 2, 2, title='png')
ax2.set_axis_off()
ind = ax2.imshow(png.T, interpolation='nearest', origin='lower')
ax3 = fig.add_subplot(2, 2, 3, title='msd')
ax3.set_axis_off()
ind = ax3.imshow(msd.T, interpolation='nearest', origin='lower', vmin=0)
plt.savefig('SHORE_maps_2.png')

"""
.. figure:: SHORE_maps.png
   :align: center

   **rtop and msd calculated using the SHORE model**.


.. [Descoteaux2011] Descoteaux M. et. al , "Multiple q-shell diffusion
					propagator imaging", Medical Image Analysis, vol 15,
					No. 4, p. 603-621, 2011.

.. [Ozarslan2013] Ozarslan E. et. al, "Mean apparent propagator (MAP) MRI: A novel
        			diffusion imaging method for mapping tissue microstructure",
        			NeuroImage, 2013

.. [Wu2007] Wu Y. et. al, "Hybrid diffusion imaging", NeuroImage, vol 36,
        	p. 617-629, 2007.

.. [Wu2008] Wu Y. et. al, "Computation of Diffusion Function Measures
			in q -Space Using Magnetic Resonance Hybrid Diffusion Imaging",
			IEEE TRANSACTIONS ON MEDICAL IMAGING, vol. 27, No. 6, p. 858-865,
			2008.

.. include:: ../links_names.inc

"""
