#
import numpy as np
import os
import sys
import shutil
import subprocess
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import vtk

import taichi as ti
from pyevtk.vtk import VtkFile, VtkRectilinearGrid, VtkUnstructuredGrid, VtkVertex

# remove everything in dir


def remove_everything_in(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def write_vtks(w_numpy, smoke_numpy, outdir, i):
    data = w_numpy.squeeze()
    smoke_data = smoke_numpy.squeeze()
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order="F"), deep=True)
    vtkDataArray.SetName("vorticity")
    imageData.GetPointData().SetScalars(vtkDataArray)

    smokeDataArray = numpy_support.numpy_to_vtk(
        smoke_data.ravel(order="F"), deep=True)
    smokeDataArray.SetName("smoke")
    imageData.GetPointData().AddArray(smokeDataArray)

    # write to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:04d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()


def write_vtks_4channel_smoke(w_numpy, smoke_numpy, outdir, i):
    data = w_numpy.squeeze()
    # smoke_data = smoke_numpy.squeeze()
    imageData = vtk.vtkImageData()

    imageData.SetDimensions(data.shape)
    imageData.SetOrigin(0, 0, 0)
    imageData.SetSpacing(1, 1, 1)

    vtkDataArray = numpy_support.numpy_to_vtk(data.ravel(order="F"), deep=True)
    vtkDataArray.SetName("vorticity")
    imageData.GetPointData().SetScalars(vtkDataArray)

    smoke_vector_data = smoke_numpy.reshape(-1, 4, order="F")
    smokeDataArray = numpy_support.numpy_to_vtk(smoke_vector_data, deep=True)
    smokeDataArray.SetNumberOfComponents(4)
    smokeDataArray.SetName("smoke")
    imageData.GetPointData().AddArray(smokeDataArray)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outdir, "field_{:04d}.vti".format(i)))
    writer.SetInputData(imageData)
    writer.Write()


def split_array(data):
    x, y, z = np.copy(data[:, 0]), np.copy(data[:, 1]), np.copy(data[:, 2])
    return x, y, z


def write_to_vtks_lp(pos, scalar_data, vector_data, file_path):
    # here points is regarded as unstructure-grid
    w = VtkFile(file_path, VtkUnstructuredGrid)
    npoints = pos.shape[0]

    w.openGrid()
    w.openPiece(ncells=npoints, npoints=npoints)

    # add points
    w.openElement("Points")
    x, y, z = split_array(pos)
    w.addData("points", (x, y, z))
    w.closeElement("Points")

    # add cell, now the cell is meaning less
    w.openElement("Cells")
    # index of last node in each cell
    offsets = np.arange(start=1, stop=npoints + 1, dtype='int32')
    # each point is only connected to itself
    connectivity = np.arange(npoints, dtype='int32')
    cell_types = np.empty(npoints, dtype='uint8')
    cell_types[:] = VtkVertex.tid
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    # add data
    scalar_keys = sorted(list(scalar_data.keys()))
    vector_keys = sorted(list(vector_data.keys()))
    if (len(scalar_keys) > 0 and len(vector_keys) > 0):
        w.openData("Point", scalars=scalar_keys[0], vectors=vector_keys[0])
    elif (len(scalar_keys) > 0):
        w.openData("Point", scalars=scalar_keys[0])
    elif (len(vector_keys) > 0):
        w.openData("Point", vectors=vector_keys[0])
    else:
        w.openData("Point", scalars="scalar")
    for key in scalar_keys:
        data = scalar_data[key]
        w.addData(key, data)
    for key in vector_keys:
        data = vector_data[key]
        vx, vy, vz = split_array(data)
        w.addData(key, (vx, vy, vz))
    w.closeData("Point")
    w.closePiece()
    w.closeGrid()

    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)
    for key in scalar_keys:
        data = scalar_data[key]
        w.appendData(data)
    for key in vector_keys:
        data = vector_data[key]
        vx, vy, vz = split_array(data)
        w.appendData((vx, vy, vz))
    w.save()
