#
# Copyright (c) Safran, 2024
# All rights reserved.
#

import numpy as np
import scipy.spatial as spatial
import bisect, copy, time
import os, json
from tqdm import tqdm
import triangle as tr
import multiprocess as mp
from functools import partial

from Muscat.Containers import MeshCreationTools as MCT
from Muscat.Containers import MeshGraphTools as MGT
from Muscat.Containers.NativeTransfer import NativeTransfer
from Muscat.Containers import MeshModificationTools as MMT
from Muscat.Containers import MeshFieldOperations as MFO
from Muscat.Containers.Filters import FilterObjects as FO
from Muscat.Containers.Filters import FilterOperators as FOp

from Muscat.FE.FETools import PrepareFEComputation
from Muscat.FE.Fields.FEField import FEField
from Muscat.ImplicitGeometry.ImplicitGeometryObjects import  ImplicitGeometrySphere

import pyvista as pv

import Muscat.Containers.ElementsDescription as ED
import Muscat.Containers.MeshInspectionTools as MIT

from Muscat.Helpers import Profiler as P

from airfrans.simulation import Simulation


p0 = np.array([1.,0.,0.5])
nb_points_add_ext_boundary = 100

out_field_names = ['UX', 'UY', 'p', 'nut']

h = 1.718
BBox = (np.array([-2.26, -h,  0.5]), np.array([4.33, h, 0.5]))
PBBox = [[BBox[0][0], BBox[0][1]], [BBox[1][0], BBox[0][1]], [BBox[1][0], BBox[1][1]], [BBox[0][0], BBox[1][1]]]


def ComputeSillageNodeIds(mesh):

    cell_ids_field = mesh.elemFields['cell_ids']
    assert(list(cell_ids_field) == sorted(cell_ids_field))
    diff = cell_ids_field[1:] - cell_ids_field[:-1]
    jumps_pos = np.where(diff>2000)[0]
    assert(len(jumps_pos)==5)

    jumps_pos = np.concatenate(([0], jumps_pos, [len(cell_ids_field)-1]))
    cluster_min_val = []
    cluster_max_val = []
    for i_jump in range(1,len(jumps_pos)):
        cluster_min_val.append(cell_ids_field[jumps_pos[i_jump-1]+1])
        cluster_max_val.append(cell_ids_field[jumps_pos[i_jump]])
    cluster_max_val.append(np.max(cell_ids_field)+1)
    for i_zone,(umin,umax) in enumerate(zip(cluster_min_val, cluster_max_val)):
        elFilter = FO.ElementFilter(eMask=(np.logical_and(umin<=cell_ids_field, cell_ids_field<=umax)))
        ids = elFilter.GetIdsToTreat(mesh, "quad4")
        mesh.elements["quad4"].GetTag(f'zone_{i_zone}').SetIds(ids)

    sillage_node_filter_1 = FO.NodeFilter(eTag = ["zone_0"])
    sillage_node_filter_2 = FO.NodeFilter(eTag = ["zone_1"])
    sillage_node_filter   = FOp.IntersectionFilter(filters=[sillage_node_filter_1, sillage_node_filter_2])
    sillage_node_ids      = sillage_node_filter.GetNodesIndices(mesh)

    for i_zone in range(6):
        mesh.elements["quad4"].tags.DeleteTags([f'zone_{i_zone}'])

    return sillage_node_ids



def ExtractPathFromMeshOfBars(mesh, startingClosestToPoint, trigo_dir = True):

    nodeGraph0Airfoild = MGT.ComputeNodeToNodeGraph(mesh, dimensionality=1)
    nodeGraphAirfoild = [list(nodeGraph0Airfoild[i].keys()) for i in range(nodeGraph0Airfoild.number_of_nodes())]

    tree = spatial.KDTree(mesh.nodes)
    _, indicesTrailEdge = tree.query([startingClosestToPoint], k=1)

    p1init = indicesTrailEdge[0]

    temp1=mesh.nodes[nodeGraphAirfoild[p1init][0]][1]
    temp2=mesh.nodes[nodeGraphAirfoild[p1init][1]][1]

    if trigo_dir:
        condition = temp1 > temp2
    else:
        condition = temp1 < temp2

    if condition:
        p2 = nodeGraphAirfoild[p1init][0]
    else:
        p2 = nodeGraphAirfoild[p1init][1]

    p1 = p1init
    path = [p1, p2]
    while p2 != p1init:
        p2save = p2
        tempArray = np.asarray(nodeGraphAirfoild[p2])
        p2 = tempArray[tempArray!=p1][0]
        p1 = p2save
        path.append(p2)

    return path


def ExtractAirfoil(mesh, scalars):

    efAirfoil = FO.ElementFilter(elementType=ED.Bar_2, eTag=["Airfoil"])
    airfoilMesh = MIT.ExtractElementsByElementFilter(mesh, efAirfoil)

    path = ExtractPathFromMeshOfBars(airfoilMesh, p0)

    tree = spatial.KDTree(airfoilMesh.nodes[path])
    _, indicesLeadEdge = tree.query([[0.,0.,0.5]], k=1)

    indices_extrado = path[:indicesLeadEdge[0]+1]
    indices_intrado = path[indicesLeadEdge[0]:]

    indices_airfoil = [indices_extrado, indices_intrado]

    nodes_extrado = mesh.nodes[indices_extrado]
    nodes_intrado = mesh.nodes[indices_intrado]

    nodes_airfoil = [nodes_extrado, nodes_intrado]

    return indices_airfoil, nodes_airfoil



def computeAirfoilCurvAbscissa(airfoil):

    indices_airfoil = airfoil[0]
    nodes_airfoil = airfoil[1]

    curv_abscissa = []
    for i in range(2):
        local_curv_abscissa = np.zeros(len(indices_airfoil[i]))
        for j in range(1,len(local_curv_abscissa)):
            local_curv_abscissa[j] = local_curv_abscissa[j-1] + np.linalg.norm(nodes_airfoil[i][j]-nodes_airfoil[i][j-1])
        local_curv_abscissa /= local_curv_abscissa[-1]
        curv_abscissa.append(local_curv_abscissa)

    return curv_abscissa



def MapAirfoil(airfoil_ref, curv_abscissa_ref, curv_abscissa):

    nodes_airfoil_ref = airfoil_ref[1]
    dim_nodes = nodes_airfoil_ref[0][0].shape[0]

    mapped_airfoil = []
    for i in range(2):
        local_mapped_airfoil = np.zeros((len(curv_abscissa[i])-1, dim_nodes))
        for j in range(len(curv_abscissa[i])-1):
            index = max(bisect.bisect_right(curv_abscissa_ref[i], curv_abscissa[i][j]) - 1, 0)

            a = nodes_airfoil_ref[i][index]
            b = nodes_airfoil_ref[i][index+1]
            dl = curv_abscissa[i][j] - curv_abscissa_ref[i][index]
            dir = (b-a)/np.linalg.norm(b-a)
            local_mapped_airfoil[j] = a + dl * dir
        mapped_airfoil.append(local_mapped_airfoil)

    return mapped_airfoil



def ComputeIntersectionWithBoundingBox(p, PBBox, p0):

    delta = p - p0
    mx = PBBox[0][0]
    my = PBBox[0][1]
    Mx = PBBox[2][0]
    My = PBBox[2][1]

    val = np.argmin([np.linalg.norm(p-pb) for pb in PBBox])
    if val == 0:

        lambda_ = (PBBox[0]-p0)[1]/delta[1]
        res = p0 + lambda_*delta
        if mx < res[0] < Mx:
            return res
        lambda_ = (PBBox[0]-p0)[0]/delta[0]
        res = p0 + lambda_*delta
        assert my < res[1] < My
        return res

    elif val == 1:

        lambda_ = (PBBox[1]-p0)[0]/delta[0]
        res = p0 + lambda_*delta
        if my < res[1] < My:
            return res
        lambda_ = (PBBox[1]-p0)[1]/delta[1]
        res = p0 + lambda_*delta
        assert mx < res[0] < Mx
        return res

    elif val == 2:

        lambda_ = (PBBox[1]-p0)[0]/delta[0]
        res = p0 + lambda_*delta
        if my < res[1] < My:
            return res
        lambda_ = (PBBox[2]-p0)[1]/delta[1]
        res = p0 + lambda_*delta
        assert mx < res[0] < Mx
        return res

    elif val == 3:

        lambda_ = (PBBox[2]-p0)[1]/delta[1]
        res = p0 + lambda_*delta
        if mx < res[0] < Mx:
            return res
        lambda_ = (PBBox[0]-p0)[0]/delta[0]
        res = p0 + lambda_*delta
        assert my < res[1] < My
        return res



def GetFieldTransferOpCppStep1(inputField, nbThreads = None):

    method="Interp/Clamp"

    nt = NativeTransfer()

    if nbThreads is not None:
        nt.SetMaxConcurrency(nbThreads)

    nt.SetTransferMethod(method)
    defaultOptions = {"usePointSearch": True,
                    "useElementSearch": False,
                    "useElementSearchFast": False,
                    "useEdgeSearch": True,
                    }

    options = {}

    defaultOptions.update(options)

    dispatch = {"usePointSearch": nt.SetUsePointSearch,
                "useElementSearch": nt.SetUseElementSearch,
                "useElementSearchFast": nt.SetUseElementSearchFast,
                "useEdgeSearch": nt.SetUseEdgeSearch,
                "DifferentialOperator": nt.SetDifferentialOperator,
                }

    for k, v in defaultOptions.items():
        if k in dispatch.keys():
            dispatch[k](v)
        else:
            raise RuntimeError(f"Option {k} not valid")

    nt.SetSourceFEField(inputField, None)

    return nt



def GetFieldTransferOpCppStep2(nt, targetPoints):

    nt.SetTargetPoints(targetPoints)

    nt.Compute()
    op = nt.GetOperator()
    status = nt.GetStatus()
    return op, status



def TruncatedSVDSymLower(matrix, epsilon = None, nbModes = None):

    if epsilon != None and nbModes != None:
        raise("cannot specify both epsilon and nbModes")

    eigenValues, eigenVectors = np.linalg.eigh(matrix, UPLO="L")

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    if nbModes == None:
        if epsilon == None:
            nbModes  = matrix.shape[0]
        else:
            nbModes = 0
            bound = (epsilon ** 2) * eigenValues[0]
            for e in eigenValues:
                if e > bound:
                    nbModes += 1
            id_max2 = 0
            bound = (1 - epsilon ** 2) * np.sum(eigenValues)
            temp = 0
            for e in eigenValues:
                temp += e
                if temp < bound:
                    id_max2 += 1

            nbModes = max(nbModes, id_max2)

    if nbModes > matrix.shape[0]:
        print("nbModes taken to max possible value of "+str(matrix.shape[0])+" instead of provided value "+str(nbModes))
        nbModes = matrix.shape[0]

    index = np.where(eigenValues<0)
    if len(eigenValues[index])>0:
        if index[0][0]<nbModes:
            print("removing numerical noise from eigenvalues, nbModes is set to "+str(index[0][0])+" instead of "+str(nbModes))
            nbModes = index[0][0]

    return eigenValues[0:nbModes], eigenVectors[:, 0:nbModes]



def snapshotsPOD_fit_transform(snapshots, correlationOperator, nbModes):

    numberOfSnapshots = snapshots.shape[0]
    numberOfDofs = snapshots.shape[1]
    correlationMatrix = np.zeros((numberOfSnapshots,numberOfSnapshots))
    matVecProducts = np.zeros((numberOfDofs,numberOfSnapshots))
    for i, snapshot1 in enumerate(snapshots):
        matVecProduct = correlationOperator.dot(snapshot1)
        matVecProducts[:,i] = matVecProduct
        for j, snapshot2 in enumerate(snapshots):
            if j <= i and j < numberOfSnapshots:
                correlationMatrix[i, j] = np.dot(matVecProduct, snapshot2)

    eigenValuesRed, eigenVectorsRed = TruncatedSVDSymLower(correlationMatrix, nbModes = nbModes)

    nbePODModes = eigenValuesRed.shape[0]
    print("truncature =", eigenValuesRed[-1]/eigenValuesRed[0])

    changeOfBasisMatrix = np.zeros((nbePODModes,numberOfSnapshots))
    for j in range(nbePODModes):
        changeOfBasisMatrix[j,:] = eigenVectorsRed[:,j]/np.sqrt(eigenValuesRed[j])

    reducedOrderBasis = np.dot(changeOfBasisMatrix,snapshots)
    generalizedCoordinates = np.dot(reducedOrderBasis, matVecProducts).T
    return reducedOrderBasis, generalizedCoordinates


def pretreat_sample(benchmark_path, folder, n_threads = 1):

    ###################
    # Read the raw data
    ###################

    simulation = Simulation(benchmark_path, folder)

    pointFields = [simulation.velocity[:,0], simulation.velocity[:,1], simulation.pressure, simulation.nu_t]

    force_C = simulation.force_coefficient()
    assert(len(force_C) == 2)
    C_D = np.float64(force_C[0][0])
    C_L = np.float64(force_C[1][0])

    scalars = [float(simulation.inlet_velocity), float(simulation.angle_of_attack), C_D, C_L]

    mesh_0 = pv.read(os.path.join(benchmark_path, folder, folder+'_internal.vtu'))

    ppp = np.hstack((simulation.position, 0.5*np.ones(simulation.position.shape[0]).reshape((-1,1))))
    mesh = MCT.CreateMeshOf(ppp, mesh_0.cell_connectivity.reshape((-1,4)), elemName=ED.Quadrangle_4)
    mesh.elemFields['cell_ids'] = mesh_0.cell_data['cell_ids']

    # Compute sillage_node_ids
    sillage_node_ids = ComputeSillageNodeIds(mesh)
    mesh.GetNodalTag("Sillage").AddToTag(sillage_node_ids)

    mesh.nodeFields = {}
    mesh.elemFields = {}

    ###################
    # Convert to triangles
    ###################
    MCT.MeshToSimplex(mesh)
    mesh.elements['tri3'].connectivity = mesh.elements['tri3'].connectivity[:,[0,2,1]]
    mesh.ConvertDataForNativeTreatment()

    ###################
    # Compute the skin of the mesh (containing the external boundary and the airfoil boundary)
    ###################
    MMT.ComputeSkin(mesh, md = 2, inPlace = True)

    ###################
    # Extract ids of the bar elements corresponding to the airfoil
    ###################
    ff1 = FO.ElementFilter(zone = lambda p: (-p[:,0]-1.99))
    ff2 = FO.ElementFilter(zone = lambda p: (p[:,0]-3.99))
    ff3 = FO.ElementFilter(zone = lambda p: (-p[:,1]-1.49))
    ff4 = FO.ElementFilter(zone = lambda p: (p[:,1]-1.49))
    efAirfoil = FOp.IntersectionFilter(filters=[ff1, ff2, ff3, ff4])
    airfoil_ids = efAirfoil.GetIdsToTreat(mesh, "bar2")

    ###################
    # Preparations
    ###################
    # Displace node at the end of the sillage to the bounding box
    ext_bound = np.setdiff1d(mesh.elements["bar2"].GetTag("Skin").GetIds(), airfoil_ids)
    mesh.elements["bar2"].GetTag("External_boundary").SetIds(ext_bound)
    nfExtBoundary = FO.NodeFilter(eTag = "External_boundary")
    nodeIndexExtBoundary  = nfExtBoundary.GetNodesIndices(mesh)

    # Prepare FE interpolation operator
    space, numberings, _, _ = PrepareFEComputation(mesh)
    inputFEField = FEField(name="dummy", mesh=mesh, space=space, numbering=numberings[0])

    amax = np.argmax(mesh.nodes[sillage_node_ids, 0])
    bound_end_sillage = ImplicitGeometrySphere(radius=0.4, center=mesh.nodes[amax,:])
    nnFilter= FO.NodeFilter(zone = bound_end_sillage)
    nf = FOp.IntersectionFilter(filters=[nfExtBoundary, nnFilter])
    bound_end_sillage_nodes_ids = nf.GetNodesIndices(mesh)
    mesh.GetNodalTag("Bound_end_sillage").AddToTag(bound_end_sillage_nodes_ids)

    mesh.GetNodalTag("rest_out_bound").AddToTag(np.setdiff1d(nodeIndexExtBoundary, bound_end_sillage_nodes_ids))
    mesh.GetNodalTag("External_boundary").AddToTag(nodeIndexExtBoundary)

    for i in bound_end_sillage_nodes_ids:
        mesh.nodes[i,:2] = ComputeIntersectionWithBoundingBox(mesh.nodes[i,:2], PBBox, p0[:2])
    mesh.ConvertDataForNativeTreatment()

    # Compute intermediate partial boundary meshmesh
    amin = np.argmin(mesh.nodes[bound_end_sillage_nodes_ids,1])
    amax = np.argmax(mesh.nodes[bound_end_sillage_nodes_ids,1])
    mesh.GetNodalTag("extreme_end_sillage_nodes").AddToTag(np.array([bound_end_sillage_nodes_ids[amin], bound_end_sillage_nodes_ids[amax]]))

    ext_p0 = mesh.nodes[bound_end_sillage_nodes_ids[amax]]
    ext_p1 = mesh.nodes[bound_end_sillage_nodes_ids[amin]]

    ef_rest_out_bound_mesh = FO.ElementFilter(elementType=ED.Bar_2, nTag = ["rest_out_bound", "extreme_end_sillage_nodes"])
    rest_out_bound_mesh = MIT.ExtractElementsByElementFilter(mesh, ef_rest_out_bound_mesh)
    MMT.CleanLonelyNodes(rest_out_bound_mesh)

    # Clean intermediate tags
    mesh.elements["bar2"].tags.DeleteTags(["Skin", "External_boundary"])
    mesh.nodesTags.DeleteTags(["rest_out_bound", "extreme_end_sillage_nodes", "External_boundary"])

    ###################
    # Add bar elements and tag for the airfoil only
    ###################
    mesh.elements["bar2"].connectivity = mesh.elements["bar2"].connectivity[airfoil_ids,:]
    mesh.ConvertDataForNativeTreatment()
    mesh.elements["bar2"].cpt = len(airfoil_ids)
    mesh.elements["bar2"].GetTag("Airfoil").SetIds(np.arange(len(airfoil_ids)))
    mesh.elements["bar2"].originalIds = np.arange(len(airfoil_ids))

    ###################
    # Add node tag for the intrado and extrado
    ###################
    nfAirfoil = FO.NodeFilter(eTag = "Airfoil")
    nodeIndexAirfoil = nfAirfoil.GetNodesIndices(mesh)
    mesh.GetNodalTag("Airfoil").AddToTag(nodeIndexAirfoil)

    airfoil = ExtractAirfoil(mesh, scalars)
    indices_extrado = airfoil[0][0]
    indices_intrado = airfoil[0][1]
    mesh.GetNodalTag("Extrado").AddToTag(indices_extrado)
    mesh.GetNodalTag("Intrado").AddToTag(indices_intrado)

    efExtrado = FO.ElementFilter(nTag = "Extrado")
    efIntrado = FO.ElementFilter(nTag = "Intrado")
    mesh.elements["bar2"].GetTag("Extrado").SetIds(efExtrado.GetIdsToTreat(mesh, "bar2"))
    mesh.elements["bar2"].GetTag("Intrado").SetIds(efIntrado.GetIdsToTreat(mesh, "bar2"))

    ###################
    # Create the mesh between the boundary box and the external boundary of the input mesh
    ###################

    points_to_add = [ext_p0[:2]]
    nb_points_to_add_top = int(nb_points_add_ext_boundary*(PBBox[2][1] - ext_p0[1])/(PBBox[2][1] - PBBox[1][1]))
    nb_points_to_add_down = int(nb_points_add_ext_boundary*(ext_p1[1] - PBBox[1][1])/(PBBox[2][1] - PBBox[1][1]))

    for i in range(nb_points_to_add_top):
        points_to_add.append(ext_p0[:2]+((i+1)/nb_points_to_add_top)*(PBBox[2]-ext_p0[:2]))
    for i in range(nb_points_add_ext_boundary):
        Delta_P = np.array(PBBox[3]) - np.array(PBBox[2])
        points_to_add.append(PBBox[2] + (i+1)/(nb_points_add_ext_boundary)*Delta_P)
    for i in range(nb_points_add_ext_boundary):
        Delta_P = np.array(PBBox[0]) - np.array(PBBox[3])
        points_to_add.append(PBBox[3] + (i+1)/(nb_points_add_ext_boundary)*Delta_P)
    for i in range(nb_points_add_ext_boundary):
        Delta_P = np.array(PBBox[1]) - np.array(PBBox[0])
        points_to_add.append(PBBox[0] + (i+1)/(nb_points_add_ext_boundary)*Delta_P)
    for i in range(nb_points_to_add_down):
        points_to_add.append(PBBox[1]+((i+1)/nb_points_to_add_down)*(ext_p1[:2]-PBBox[1]))

    vert = np.vstack((rest_out_bound_mesh.nodes[:,:2], points_to_add))
    nn = vert.shape[0]
    mm = len(points_to_add)
    indices_to_add = []
    for j in range(mm-1):
        indices_to_add.append([nn-mm+j, nn-mm+(j+1)%mm])
    seg = np.vstack((rest_out_bound_mesh.elements['bar2'].connectivity, indices_to_add))

    temp_mesh = MCT.CreateMeshOf(vert, seg, 'bar2')
    MMT.CleanDoubleNodes(temp_mesh)
    MMT.CleanDoubleElements(temp_mesh)
    temp_mesh.ConvertDataForNativeTreatment()

    di = {'vertices':temp_mesh.nodes, 'segments':temp_mesh.elements['bar2'].connectivity, 'holes':[[0.5,0.]]}
    t = tr.triangulate(di, 'pc')
    total_mesh = MCT.CreateMeshOfTriangles(t['vertices'], t['triangles'])
    total_mesh.nodes = np.hstack((total_mesh.nodes, 0.5*np.ones(total_mesh.nodes.shape[0]).reshape((-1,1))))
    total_mesh.ConvertDataForNativeTreatment()

    ###################
    # Merge the meshes
    ###################
    pretreated_mesh = copy.deepcopy(mesh)
    pretreated_mesh.Merge(total_mesh)
    MMT.CleanDoubleNodes(pretreated_mesh)
    MMT.CleanDoubleElements(pretreated_mesh)
    pretreated_mesh.DeleteElemTags(["2D"])
    pretreated_mesh.ConvertDataForNativeTreatment()
    pretreated_mesh.Clean()

    ###################
    # Add External_boundary element and node tags
    ###################
    MMT.ComputeSkin(pretreated_mesh, md = 2, inPlace = True)
    ext_bound = np.setdiff1d(pretreated_mesh.elements["bar2"].GetTag("Skin").GetIds(), pretreated_mesh.elements["bar2"].GetTag("Airfoil").GetIds())
    pretreated_mesh.elements["bar2"].GetTag("External_boundary").SetIds(ext_bound)
    nfExtBound = FO.NodeFilter(eTag = "External_boundary")
    nodeIndexExtBound = nfExtBound.GetNodesIndices(pretreated_mesh)
    pretreated_mesh.GetNodalTag("External_boundary").AddToTag(nodeIndexExtBound)
    pretreated_mesh.elements["bar2"].tags.DeleteTags(["Skin"])

    ###################
    # Compute nodeFields values in the mesh
    ###################

    op, _ = MFO.GetFieldTransferOpCpp(inputFEField, pretreated_mesh.nodes, nbThreads = n_threads)

    for pfn, pf in zip(out_field_names, pointFields):
        pretreated_mesh.nodeFields[pfn] = op.dot(pf)

    size_init_mesh = mesh.GetNumberOfNodes()

    return pretreated_mesh, size_init_mesh, scalars



def morph_sample(mesh_1, scalars_1, airfoil_0, curv_abscissa_0):

    ##############################################################
    # Compute the mapping of the extrado and intrado
    ##############################################################

    airfoil_1 = ExtractAirfoil(mesh_1, scalars_1)
    curv_abscissa_1 = computeAirfoilCurvAbscissa(airfoil_1)
    mapped_airfoil = MapAirfoil(airfoil_0, curv_abscissa_0, curv_abscissa_1)

    indices_extrado_to_morph_1 = airfoil_1[0][0][:-1]
    indices_intrado_to_morph_1 = airfoil_1[0][1][:-1]

    ##############################################################
    # Compute the mapping of the sillage
    ##############################################################

    sillage_node_ids = mesh_1.GetNodalTag("Sillage").GetIds()
    ind_sillage_0 = sillage_node_ids[0]

    sillage_node_ids = sillage_node_ids[1:-8]

    # Compute displacement of the sillage nodes consisted of incidence angle rotation and correction wrt nut
    delta_f = mesh_1.nodes[ind_sillage_0,:2] + - p0[:2]
    angle_1 = np.arctan(delta_f[1]/delta_f[0])

    rot_matrix = np.array([[np.cos(-angle_1), -np.sin(-angle_1)], [np.sin(-angle_1), np.cos(-angle_1)]])

    l5 = len(sillage_node_ids)
    displacement_sillage = np.zeros((l5,2))

    for i, id in enumerate(sillage_node_ids):
        displacement_sillage[i,:] = p0[:2] + np.dot(rot_matrix,  mesh_1.nodes[id, :2] - p0[:2]) -  mesh_1.nodes[id, :2]

    ##############################################################
    # Compute the mapping of the outflow boundary
    ##############################################################

    indices_ext_bound_to_morph_1 = mesh_1.GetNodalTag("External_boundary").GetIds()
    nf = FO.NodeFilter(zone = lambda p: (-p[:,0] + PBBox[2][0]-0.0001))
    out_boundary_ids_1 = nf.GetNodesIndices(mesh_1)
    mesh_1.GetNodalTag("Out_boundary").AddToTag(out_boundary_ids_1)
    other_boundary_ids_1 = np.setdiff1d(indices_ext_bound_to_morph_1, out_boundary_ids_1)
    mesh_1.GetNodalTag("Other_boundary").AddToTag(other_boundary_ids_1)

    bound_end_sillage_ids = mesh_1.GetNodalTag("Bound_end_sillage").GetIds()
    l4 = len(bound_end_sillage_ids)
    displacement_out_boundary = np.zeros((l4,2))

    for i, id in enumerate(bound_end_sillage_ids):
        displacement_out_boundary[i,1] = p0[1] + np.dot(rot_matrix,  mesh_1.nodes[id, :2] - p0[:2])[1] -  mesh_1.nodes[id, 1]

    ##############################################################
    # Compute global target displacement and masks for RBF field morphing
    ##############################################################

    l1 = len(indices_extrado_to_morph_1)
    l2 = len(indices_intrado_to_morph_1)
    l3 = len(other_boundary_ids_1)

    targetDisplacement     = np.zeros((l1 + l2 + l3 + l4 + l5, 3))
    targetDisplacementMask = np.zeros((l1 + l2 + l3 + l4 + l5), dtype = int)

    targetDisplacement[:l1,:2]                        = mapped_airfoil[0][:,:2] - mesh_1.nodes[indices_extrado_to_morph_1,:2]
    targetDisplacement[l1:l1+l2,:2]                   = mapped_airfoil[1][:,:2] - mesh_1.nodes[indices_intrado_to_morph_1,:2]
    targetDisplacement[l1+l2:l1+l2+l3,:2]             = np.zeros((l3,2))
    targetDisplacement[l1+l2+l3:l1+l2+l3+l4,:2]       = displacement_out_boundary
    targetDisplacement[l1+l2+l3+l4:l1+l2+l3+l4+l5,:2] = displacement_sillage

    targetDisplacementMask[:l1]                        = indices_extrado_to_morph_1
    targetDisplacementMask[l1:l1+l2]                   = indices_intrado_to_morph_1
    targetDisplacementMask[l1+l2:l1+l2+l3]             = other_boundary_ids_1
    targetDisplacementMask[l1+l2+l3:l1+l2+l3+l4]       = bound_end_sillage_ids
    targetDisplacementMask[l1+l2+l3+l4:l1+l2+l3+l4+l5] = sillage_node_ids

    ##############################################################
    # Compute the morphing
    ##############################################################

    # RBF morphing
    mesh_1_nodes = mesh_1.nodes.copy()
    morphed_nodes = MMT.Morphing(mesh_1, targetDisplacement, targetDisplacementMask, radius=None)
    mesh_1.nodes = morphed_nodes

    # final clean for out_bound out of sillage
    top_ind = np.argmax(mesh_1.nodes[out_boundary_ids_1,1])
    down_ind = np.argmin(mesh_1.nodes[out_boundary_ids_1,1])
    ind_to_clean = np.setdiff1d(out_boundary_ids_1, bound_end_sillage_ids)
    mesh_1.nodes[out_boundary_ids_1[top_ind],:2] = PBBox[2]
    mesh_1.nodes[out_boundary_ids_1[down_ind],:2] = PBBox[1]
    mesh_1.nodes[ind_to_clean,0] = PBBox[1][0]
    mesh_1.ConvertDataForNativeTreatment()

    mesh_1.nodeFields['X'] = mesh_1_nodes[:,0]
    mesh_1.nodeFields['Y'] = mesh_1_nodes[:,1]

    return mesh_1



def project_sample(morphed_mesh, morphed_mesh_0):
    projected_mesh = copy.deepcopy(morphed_mesh_0)

    space_, numberings_, _, _ = PrepareFEComputation(morphed_mesh)
    inputFEField = FEField(name="dummy", mesh=morphed_mesh, space=space_, numbering=numberings_[0])

    nt = GetFieldTransferOpCppStep1(inputFEField, 1)
    FE_interpolation_op, _ = GetFieldTransferOpCppStep2(nt, morphed_mesh_0.nodes)

    for pfn in out_field_names + ['X', 'Y']:
        projected_mesh.nodeFields[pfn] = FE_interpolation_op.dot(morphed_mesh.nodeFields[pfn])

    return projected_mesh, FE_interpolation_op


def pretreat_morph_and_project_mesh(i_sample, benchmark_path, manifest, morphed_mesh_0, airfoil_0, curv_abscissa_0):


    ###################
    ## 1) Pretreat data
    ###################
    pretreated_mesh, size_init_mesh, scalars = pretreat_sample(benchmark_path, manifest[i_sample])

    ###################
    ## 2) Morph data
    ###################
    morphed_mesh = morph_sample(pretreated_mesh, scalars, airfoil_0, curv_abscissa_0)

    ###################
    ## 3) Project data
    ###################
    projected_mesh, FE_interpolation_op = project_sample(morphed_mesh, morphed_mesh_0)

    return [projected_mesh, size_init_mesh, scalars, FE_interpolation_op, morphed_mesh.nodes]



def get_dataset_name(dataset, train):

    taskk = 'full' if dataset._task == 'scarce' and not train else dataset._task
    split = 'train' if train else 'test'

    return taskk + '_' + split


def reynolds_filter(dataset):
    simulation_names=dataset.extra_data["simulation_names"]
    reynolds=np.array([float(name.split('_')[2])/1.56e-5 for name,numID in simulation_names])
    simulation_indices=np.where((reynolds>3e6) & (reynolds<5e6))[0]
    filtered_reynolds=reynolds[simulation_indices]
    return filtered_reynolds


def safran_process_dataset(dataset, train, benchmark_path):

    dataset_name = get_dataset_name(dataset, train)

    with open(os.path.join(benchmark_path, 'manifest.json'), 'r') as f:
        manifest_full = json.load(f)

    if train == True:
        reynolds = np.array([float(manifest_full[dataset_name][i].split('_')[2])/1.56e-5 for i in range(len(manifest_full[dataset_name]))])
        simulation_indices = np.where((reynolds>3e6) & (reynolds<5e6))[0]

        filtered_reynolds = [reynolds[i] for i in simulation_indices]
        filtered_reynolds_ref = reynolds_filter(dataset)
        assert np.allclose(filtered_reynolds, filtered_reynolds_ref) == True

        manifest = [manifest_full[dataset_name][i] for i in simulation_indices]

    else:
        manifest = manifest_full[dataset_name]

    nb_samples = len(manifest)

    n_cores = mp.cpu_count()
    print(f"number of cores: {n_cores}, nb_samples: {nb_samples}")
    n_parallel_tasks = min(nb_samples, n_cores)


    start = time.time()
    # Get first pretreated training mesh
    folder_0 = manifest_full['scarce_train'][1]
    pretreated_mesh_0, _, scalars_0 = pretreat_sample(benchmark_path, folder_0, n_threads = n_parallel_tasks)

    # Pretreat airfoil of first training mesh
    airfoil_0 = ExtractAirfoil(pretreated_mesh_0, scalars_0)
    curv_abscissa_0 = computeAirfoilCurvAbscissa(airfoil_0)

    # Morph airfoil of first training mesh
    morphed_mesh_0 = morph_sample(pretreated_mesh_0, scalars_0, airfoil_0, curv_abscissa_0)
    print(f"duration pretreat_and_morph_mesh_0 = {int(time.time() - start)} s")

    start = time.time()
    print("Treating dataset "+dataset_name)
    with mp.Pool(n_parallel_tasks) as pool:
        results = list(tqdm(
            pool.imap(partial(pretreat_morph_and_project_mesh, benchmark_path = benchmark_path, manifest = manifest, \
                    morphed_mesh_0 = morphed_mesh_0, airfoil_0 = airfoil_0, curv_abscissa_0 = curv_abscissa_0), \
                    range(nb_samples)), total = nb_samples, disable = False))
    print(f"duration pretreat_morph_and_project_mesh = {int(time.time() - start)} s")

    start = time.time()
    space_, numberings_, _, _ = PrepareFEComputation(morphed_mesh_0)
    inputFEField_0 = FEField(name="dummy", mesh=morphed_mesh_0, space=space_, numbering=numberings_[0])
    nt_0 = GetFieldTransferOpCppStep1(inputFEField_0, nbThreads = n_parallel_tasks)

    for i_sample in range(nb_samples):
        morphed_mesh_nodes = results[i_sample][-1]
        results[i_sample][-1] = GetFieldTransferOpCppStep2(nt_0, morphed_mesh_nodes)[0]
    print(f"duration inverse FE operator computation = {int(time.time() - start)} s")

    return results



def parse_data(path_to_simulations, data, boundary_thickness):

    clouds, scalars, fields, fields_airfoil, forward_fe_op, inverse_fe_op = [], [], [], [], [], []

    mesh_0 = data[0][0]

    with open(os.path.join(path_to_simulations, 'manifest.json'), 'r') as f:
        manifest_full = json.load(f)
    folder_0 = manifest_full['scarce_train'][1]

    mesh_00 = pv.read(os.path.join(path_to_simulations, folder_0, folder_0+'_internal.vtu'))

    airfoil_indices = np.arange(mesh_00.n_points)[mesh_00.point_data['implicit_distance'] > -boundary_thickness]
    mesh_0.GetNodalTag("boundary_layer").AddToTag(airfoil_indices)

    sizes_init_meshes = []
    for item in data:
        mesh, size_init_mesh, all_scalars, fe_op, inv_fe_op  = item

        cloud = np.stack([mesh.nodeFields["X"], mesh.nodeFields["Y"]], axis=1)

        field = np.stack([mesh.nodeFields["UX"], mesh.nodeFields["UY"], mesh.nodeFields["p"].squeeze(), mesh.nodeFields["nut"].squeeze()], axis=1)
        field_airfoil = np.stack([mesh.nodeFields["UX"][airfoil_indices], mesh.nodeFields["UY"][airfoil_indices], mesh.nodeFields["p"].squeeze()[airfoil_indices], mesh.nodeFields["nut"].squeeze()[airfoil_indices]], axis=1)
        clouds.append(cloud)
        fields.append(field)
        fields_airfoil.append(field_airfoil)
        scalars.append(np.array(all_scalars))
        forward_fe_op.append(fe_op)
        inverse_fe_op.append(inv_fe_op)
        sizes_init_meshes.append(size_init_mesh)
    clouds = np.stack(clouds)
    clouds = clouds.reshape(clouds.shape[0], -1)
    fields = np.stack(fields)
    fields_airfoil = np.stack(fields_airfoil)
    scalars = np.stack(scalars)

    return mesh_0, clouds, fields, fields_airfoil, airfoil_indices, scalars, sizes_init_meshes, forward_fe_op, inverse_fe_op
