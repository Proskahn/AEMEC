#----------------------------------------------------------------------#
# Solver      |   openFuelCell                                         #
# OpenFOAM    |   OpenFOAM-v1906 or newer (ESI)                        #
#----------------------------------------------------------------------#
# Source code |   https://github.com/openFuelCell2/openFuelCell2       #
# Update from |   14.09.2023                                           #
#----------------------------------------------------------------------#

clear:
	(./Allclean)

################################################################################

mesh:
	(cp -rf ./system/controlDict.mesh ./system/controlDict)
	(blockMesh)
	(renumberMesh -overwrite)
	(./preprocessing.sh | tee log.pre)
	(cp -rf ./system/controlDict.run ./system/controlDict)
	(./check_case.py)

salomeMesh:
	(cp -rf ./system/controlDict.mesh ./system/controlDict)
	(salome -t salomeMesh.py)
	(ideasUnvToFoam salome.unv)
	(renumberMesh -overwrite)
	(./preprocessing.sh | tee log.pre)
	(cp -rf ./system/controlDict.run ./system/controlDict)
	(./check_case.py)

check:
	(./check_case.py)

################################################################################

decompose:
	(decomposeParID -region anode -coordinate '((0.00 -0.001 0) (0.04 0.001 0))')
	(decomposeParID -region cathode -coordinate '((0.00 -0.001 0) (0.04 0.001 0))')
	(decomposeParID -region electrolyte -coordinate '((0.00 -0.001 0) (0.04 0.001 0))')
	(decomposeParID -region interconnect -coordinate '((0.00 -0.001 0) (0.04 0.001 0))')
	(decomposeParID -region phiECathode -coordinate '((0.00 -0.001 0) (0.04 0.001 0))')
	(decomposeParID -region phiEAnode -coordinate '((0.00 -0.001 0) (0.04 0.001 0))')
	(decomposeParID -region phiAnion -coordinate '((0.00 -0.001 0) (0.04 0.001 0))')
	(decomposeParID -coordinate '((0.00 -0.001 0) (0.04 0.001 0))')

################################################################################

parallel:
	(./parallel.sh | tee log.parallel);

################################################################################

run: check
	(mpirun -np 4 openFuelCell -parallel -fileHandler collated | tee log.run);

################################################################################

plot:
	(gnuplot ResidualPlot)

################################################################################

srun: check
	(openFuelCell | tee log.run);

################################################################################

reconstruct:
	(reconstructParMesh);
	(reconstructParMesh -region anode 					-constant -fileHandler collated);
	(reconstructParMesh -region cathode 					-constant -fileHandler collated);
	(reconstructParMesh -region electrolyte 	-constant -fileHandler collated);
	(reconstructParMesh -region interconnect 	-constant -fileHandler collated);
	(reconstructParMesh -region phiECathode 				-constant -fileHandler collated);
	(reconstructParMesh -region phiEAnode 				-constant -fileHandler collated);
	(reconstructParMesh -region phiAnion				 	-constant -fileHandler collated);
	(reconstructPar -fileHandler collated);
	(reconstructPar -region anode -fileHandler collated);
	(reconstructPar -region cathode -fileHandler collated);
	(reconstructPar -region electrolyte -fileHandler collated);
	(reconstructPar -region interconnect -fileHandler collated);
	(reconstructPar -region phiECathode -fileHandler collated);
	(reconstructPar -region phiEAnode -fileHandler collated);
	(reconstructPar -region phiAnion -fileHandler collated);

################################################################################

view:
	(foamToVTK -latestTime);
	(foamToVTK -latestTime -region anode);
	(foamToVTK -latestTime -region cathode);
	(foamToVTK -latestTime -region electrolyte);
	(foamToVTK -latestTime -region interconnect);
	(foamToVTK -latestTime -region phiECathode);
	(foamToVTK -latestTime -region phiEAnode);
	(foamToVTK -latestTime -region phiAnion);

################################################################################

viewAll:
	(foamToVTK);
	(foamToVTK -region anode);
	(foamToVTK -region cathode);
	(foamToVTK -region electrolyte);
	(foamToVTK -region interconnect);
	(foamToVTK -region phiECathode);
	(foamToVTK -region phiEAnode);
	(foamToVTK -region phiAnion);

################################################################################
