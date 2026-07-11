#!/bin/bash

#----------------------------------------------------------------------#
# Solver      |   openFuelCell                                         #
# OpenFOAM    |   OpenFOAM-v1906 or newer (ESI)                        #
#----------------------------------------------------------------------#
# Source code |   https://github.com/openFuelCell2/openFuelCell2       #
# Update from |   14.09.2023                                           #
#----------------------------------------------------------------------#

## edit system/decomposeParDict for the desired decomposition
## set environment variable NPROCS to number of processors.
##     e.g., export NPROCS=2
## make mesh
##
## then:

echo "NPROCS = " $NPROCS

cp system/controlDict.mesh system/controlDict

decomposePar -fileHandler collated >& log.decompose

cp system/decomposeParDict system/anode/.
cp system/decomposeParDict system/cathode/.
cp system/decomposeParDict system/electrolyte/.
cp system/decomposeParDict system/interconnect/.
cp system/decomposeParDict system/phiECathode/.
cp system/decomposeParDict system/phiEAnode/.
cp system/decomposeParDict system/phiAnion/.

decomposePar -region anode -fileHandler collated
decomposePar -region cathode -fileHandler collated
decomposePar -region electrolyte -fileHandler collated
decomposePar -region interconnect -fileHandler collated
decomposePar -region phiECathode -fileHandler collated
decomposePar -region phiEAnode -fileHandler collated
decomposePar -region phiAnion -fileHandler collated

# Step 1:
# anode/cathode/electrolyte/interconnect

mv processors$NPROCS/constant/polyMesh/cellZones processors$NPROCS/constant/polyMesh/cellZones_bk

mpirun -np $NPROCS topoSet -dict ./system/topoSetDict.afei -constant -noZero -parallel -fileHandler collated
mpirun -np $NPROCS splitMeshRegions -cellZonesOnly -parallel -fileHandler collated

# sleep to wait for files
while [! -d processors$NPROCS/1];
do
    sleep 1s
done

cp -rf processors$NPROCS/1/anode/polyMesh processors$NPROCS/constant/anode
cp -rf processors$NPROCS/1/cathode/polyMesh processors$NPROCS/constant/cathode
cp -rf processors$NPROCS/1/electrolyte/polyMesh processors$NPROCS/constant/electrolyte
cp -rf processors$NPROCS/1/interconnect/polyMesh processors$NPROCS/constant/interconnect

rm -rf processors$NPROCS/1

# Step 2:
# phiECathode, phiEAnode

rm processors$NPROCS/constant/polyMesh/cellZones

mpirun -np $NPROCS topoSet -dict ./system/topoSetDict.phiE -constant -noZero -parallel -fileHandler collated
mpirun -np $NPROCS splitMeshRegions -cellZonesOnly -parallel -fileHandler collated

# sleep to wait for files
while [! -d processors$NPROCS/1];
do
    sleep 1s
done

cp -rf processors$NPROCS/1/phiECathode/polyMesh processors$NPROCS/constant/phiECathode
cp -rf processors$NPROCS/1/phiEAnode/polyMesh processors$NPROCS/constant/phiEAnode

rm -rf processors$NPROCS/1

# Step 3:
# phiAnion

rm processors$NPROCS/constant/polyMesh/cellZones

mpirun -np $NPROCS topoSet -dict ./system/topoSetDict.phiAnion -constant -noZero -parallel -fileHandler collated
mpirun -np $NPROCS splitMeshRegions -cellZonesOnly -parallel -fileHandler collated

# sleep to wait for files
while [! -d processors$NPROCS/1];
do
    sleep 1s
done

cp -rf processors$NPROCS/1/phiAnion/polyMesh processors$NPROCS/constant/phiAnion
rm -rf processors$NPROCS/1

mv processors$NPROCS/constant/polyMesh/cellZones_bk processors$NPROCS/constant/polyMesh/cellZones

## patches:

# anode zones
mpirun -np $NPROCS topoSet -region anode -noZero -constant -parallel -fileHandler collated

# cathode zones
mpirun -np $NPROCS topoSet -region cathode -noZero -constant -parallel -fileHandler collated

# electric zones
mpirun -np $NPROCS topoSet -region phiECathode -noZero -constant -parallel -fileHandler collated
mpirun -np $NPROCS topoSet -region phiEAnode -noZero -constant -parallel -fileHandler collated
mpirun -np $NPROCS topoSet -region phiAnion -noZero -constant -parallel -fileHandler collated

rm -rf system/phi0
rm -rf system/phiAnion0
rm -rf system/phiE0
rm -rf system/phiE1

cp system/controlDict.run system/controlDict
