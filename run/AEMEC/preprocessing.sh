#!/bin/bash

#----------------------------------------------------------------------#
# Solver      |   openFuelCell                                         #
# OpenFOAM    |   OpenFOAM-v1906 or newer (ESI)                        #
#----------------------------------------------------------------------#
# Source code |   https://github.com/openFuelCell2/openFuelCell2       #
# Update from |   14.09.2023                                           #
#----------------------------------------------------------------------#

# Rename the original field to 0
rm -rf 0
cp -r 0.orig 0

SECONDS=0

topoSet -dict ./system/topoSetDict.zoneToSet -noZero -constant

# Step 1:
# anode/cathode/electrolyte/interconnect
# backup the cellZones
mv constant/polyMesh/cellZones constant/polyMesh/cellZones_bk
topoSet -dict ./system/topoSetDict.afei -noZero -constant
splitMeshRegions -cellZonesOnly

# copy
cp -r 1/anode/polyMesh constant/anode/.
cp -r 1/cathode/polyMesh constant/cathode/.
cp -r 1/electrolyte/polyMesh constant/electrolyte/.
cp -r 1/interconnect/polyMesh constant/interconnect/.

rm -rf 1

# Step 2:
# phiECathode/phiEAnode

rm constant/polyMesh/cellZones
topoSet -dict ./system/topoSetDict.phiE -noZero -constant
splitMeshRegions -cellZonesOnly

cp -r 1/phiEAnode/polyMesh constant/phiEAnode/.
cp -r 1/phiECathode/polyMesh constant/phiECathode/.

rm -rf 1
rm -rf constant/phiE0
rm -rf system/phiE0
rm -rf 0/phiE0

# Step 3:
# phiAnion

rm constant/polyMesh/cellZones
topoSet -dict ./system/topoSetDict.phiAnion -noZero -constant
splitMeshRegions -cellZonesOnly

cp -r 1/phiAnion/polyMesh constant/phiAnion/.

rm -rf 1
rm -rf constant/phiE0
rm -rf system/phiE0
rm -rf 0/phiE0

# mv back the original cell zones
mv constant/polyMesh/cellZones_bk constant/polyMesh/cellZones

topoSet -region anode
topoSet -region cathode
topoSet -region phiEAnode
topoSet -region phiECathode
topoSet -region phiAnion

# Rename the original field to 0
#rm -rf 0
#mv 0.orig 0

duration=$SECONDS
echo -e "  -Executation time for preprocessing is : $duration Sec \n\n"
