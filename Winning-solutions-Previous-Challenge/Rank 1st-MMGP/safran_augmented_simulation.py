import Muscat.Containers.ElementsDescription as ED
import Muscat.Containers.MeshInspectionTools as MIT
from Muscat.Containers import MeshModificationTools as MMT

from sklearn.preprocessing import StandardScaler, MinMaxScaler


import time
import numpy as np

from Muscat.FE import FETools as FT
from Muscat.Containers.Filters import FilterObjects as FO

from gp_torch import (
    GaussianProcessRegressor
)

from utils import (
    safran_process_dataset,
    get_dataset_name,
    parse_data,
    snapshotsPOD_fit_transform
)


import torch
device = torch.device("cpu")

num_steps = [2_000]
step_size = 1.

dataset_names = ["scarce_train", "full_test", "reynolds_test"]
output_names = ["velocity", "pressure", "turbulent_viscosity"]
pred_output_names = ["x-velocity", "y-velocity", "pressure", "turbulent_viscosity"]

nb_modes_shape_embedding_ = {}
nb_modes_shape_embedding_["velocity"] = 16
nb_modes_shape_embedding_["pressure"] = 16
nb_modes_shape_embedding_["turbulent_viscosity"] = 2

nb_modes_fields_ = {}
nb_modes_fields_["velocity"] = 12
nb_modes_fields_["pressure"] = 12
nb_modes_fields_["turbulent_viscosity"] = 2

boundary_thickness_ = 1.e-3

class AugmentedSimulator:
    def __init__(
        self,
        benchmark,
        nb_modes_shape_embedding = nb_modes_shape_embedding_,
        nb_modes_fields = nb_modes_fields_,
        boundary_thickness = boundary_thickness_
    ):

        self.nb_modes_shape_embedding = nb_modes_shape_embedding
        self.nb_modes_fields = nb_modes_fields
        self.boundary_thickness = boundary_thickness

        self.name = "AirfRANSSubmission"

        self.airfoil_indices = None
        self.sizes_meshes = {}
        self.ffo = {}
        self.iffo = {}

        self.scalar_scalers = None
        self.scalar_scalers_out = None
        self.pca_clouds = None
        self.X = {}
        self.scalerX = None

        self.pca_fields = None
        self.pca_fields_airfoil = None
        self.y = {}
        self.y_airfoil = {}

        self.y_scalers = None
        self.y_scalers_airfoil = None

        self.kmodels = None
        self.kmodels_airfoil = None

        self.path_to_simulations = benchmark.benchmark_path


    def treat_dataset(self, data, dataset_name, training):

        mesh_0, clouds, fields, fields_airfoil, airfoil_indices, scalars, sizes_meshes, ffo, iffo = parse_data(
            self.path_to_simulations, data, self.boundary_thickness
        )

        self.airfoil_indices = airfoil_indices

        self.sizes_meshes[dataset_name] = sizes_meshes
        self.ffo[dataset_name] = ffo
        self.iffo[dataset_name] = iffo

        scalars_inputs = scalars[:,:2] # use only input scalars (angle_of_attack and inlet_velocity)

        U = np.empty((fields.shape[0], 2*fields.shape[1]))
        U_airfoil = np.empty((fields_airfoil.shape[0], 2*fields_airfoil.shape[1]))
        for i in range(fields.shape[0]):
            U[i,:fields.shape[1]] = fields[i,:,0]
            U[i,fields.shape[1]:] = fields[i,:,1]
            U_airfoil[i,:fields_airfoil.shape[1]] = fields_airfoil[i,:,0]
            U_airfoil[i,fields_airfoil.shape[1]:] = fields_airfoil[i,:,1]

        if training:
            # store snapshotPOD correlation operators
            elementFilter = FO.ElementFilter()
            elementFilter.SetDimensionality(2)
            self.correlationOperator1c = FT.ComputeL2ScalarProductMatrix(mesh_0, numberOfComponents=1, elementFilter=elementFilter)
            self.correlationOperator2c = FT.ComputeL2ScalarProductMatrix(mesh_0, numberOfComponents=2, elementFilter=elementFilter)

            efAirfoil = FO.ElementFilter(elementType=ED.Triangle_3, nTag=["boundary_layer"], nTagsTreatment = "ALL_NODES")
            airfoilMesh = MIT.ExtractElementsByElementFilter(mesh_0, efAirfoil)
            MMT.CleanLonelyNodes(airfoilMesh)

            self.correlationOperator1c_airfoil = FT.ComputeL2ScalarProductMatrix(airfoilMesh, numberOfComponents=1, elementFilter=elementFilter)
            self.correlationOperator2c_airfoil = FT.ComputeL2ScalarProductMatrix(airfoilMesh, numberOfComponents=2, elementFilter=elementFilter)

            print("Dimensionality reduction for the shape embedding:")
            self.scalar_scalers = StandardScaler()
            X_scalars = self.scalar_scalers.fit_transform(scalars_inputs)
            self.scalar_scalers_out = StandardScaler()

            self.pca_clouds = []
            X_pca = []
            for name in self.nb_modes_shape_embedding.values():
                pca_c, x_pca = snapshotsPOD_fit_transform(clouds, self.correlationOperator2c, name)

                self.pca_clouds.append(pca_c)
                X_pca.append(x_pca)

            unscaled_X = [np.concatenate([xtpca, X_scalars], axis=-1) for xtpca in X_pca]
            self.scalerX = [StandardScaler() for _ in range(len(self.nb_modes_shape_embedding))]
            self.X[dataset_name] = [scx.fit_transform(xt) for scx, xt in zip(self.scalerX, unscaled_X)]

            print("Dimensionality reduction for the output fields:")
            self.pca_fields = []
            self.y_scalers = []
            self.y[dataset_name] = []

            self.pca_fields_airfoil = []
            self.y_scalers_airfoil = []
            self.y_airfoil[dataset_name] = []

            pca_f, y_pca = snapshotsPOD_fit_transform(U, self.correlationOperator2c, self.nb_modes_fields[output_names[0]])
            self.pca_fields.append(pca_f)
            y_scaler = MinMaxScaler()
            y_pca = y_scaler.fit_transform(y_pca)
            self.y_scalers.append(y_scaler)
            self.y[dataset_name].append(y_pca)

            pca_f, y_pca = snapshotsPOD_fit_transform(U_airfoil, self.correlationOperator2c_airfoil, self.nb_modes_fields[output_names[0]])
            self.pca_fields_airfoil.append(pca_f)
            y_scaler_airfoil = MinMaxScaler()
            y_pca = y_scaler_airfoil.fit_transform(y_pca)
            self.y_scalers_airfoil.append(y_scaler_airfoil)
            self.y_airfoil[dataset_name].append(y_pca)

            for i in range(1, 3):
                pca_f, y_pca = snapshotsPOD_fit_transform(fields[:, :, i+1], self.correlationOperator1c, self.nb_modes_fields[output_names[i]])
                self.pca_fields.append(pca_f)
                y_scaler = MinMaxScaler()
                y_pca = y_scaler.fit_transform(y_pca)
                self.y_scalers.append(y_scaler)
                self.y[dataset_name].append(y_pca)

                pca_f, y_pca = snapshotsPOD_fit_transform(fields_airfoil[:, :, i+1], self.correlationOperator1c_airfoil, self.nb_modes_fields[output_names[i]])
                self.pca_fields_airfoil.append(pca_f)
                y_scaler_airfoil = MinMaxScaler()
                y_pca = y_scaler_airfoil.fit_transform(y_pca)
                self.y_scalers_airfoil.append(y_scaler_airfoil)
                self.y_airfoil[dataset_name].append(y_pca)

        else:
            X_pca = [np.dot(self.pca_clouds[i], self.correlationOperator2c.dot(clouds.T)).T
                    for i in range(len(self.nb_modes_shape_embedding))]

            X_scalars = self.scalar_scalers.transform(scalars_inputs)

            unscaled_X = [np.concatenate([X_pca[i], X_scalars], axis=-1)
                    for i in range(len(self.nb_modes_shape_embedding))]

            self.X[dataset_name] = [self.scalerX[i].transform(unscaled_X[i])
                    for i in range(len(self.nb_modes_shape_embedding))]

        return None

    def process_dataset(self, dataset, training: bool) -> None:
        dataset_name = get_dataset_name(dataset, training)

        start = time.time()

        print(f"Processing dataset {dataset_name}")
        data = safran_process_dataset(dataset, training, self.path_to_simulations)
        self.treat_dataset(data, dataset_name, training)
        print(f"Dataset {dataset_name} preprocessed in {int(time.time() - start)} s")


    def train(self, train_dataset, save_path=None) -> None:

        self.process_dataset(dataset=train_dataset, training=True)

        kept_indices = np.hstack((np.arange(89), np.arange(90,103))) # removing outlier

        start = time.time()
        self.kmodels = []
        self.kmodels_airfoil = []
        for i in range(len(output_names)):
            print(f">> training {output_names[i]} (bulk and boundary_layer)")

            output_dim = self.y["scarce_train"][i].shape[-1]

            X_ = torch.tensor(self.X["scarce_train"][i][kept_indices,:], dtype=torch.float64).to(device)

            kmodel_i = []
            kmodel_airfoil_i = []

            for j in range(output_dim):

                Y_ = torch.tensor(self.y["scarce_train"][i][kept_indices, j : j + 1], dtype=torch.float64).to(device)
                Y_airfoil_ = torch.tensor(self.y_airfoil['scarce_train'][i][kept_indices, j : j + 1], dtype=torch.float64).to(device)

                kmodel = []
                kmodel_airfoil = []
                for num_step in num_steps:

                    kmodel_try = GaussianProcessRegressor(length_scale=np.ones(X_.shape[1]), noise_scale=1., amplitude_scale=1.).to(device)
                    kmodel_airfoil_try = GaussianProcessRegressor(length_scale=np.ones(X_.shape[1]), noise_scale=1., amplitude_scale=1.).to(device)

                    hist = kmodel_try.fit(X_, Y_, torch.optim.AdamW(params=kmodel_try.parameters(), lr=step_size), num_step)
                    print(f"Coord. {j} | BULK    | num_step : {num_step} | NLL : {hist[-1]}")
                    hist = kmodel_airfoil_try.fit(X_, Y_airfoil_, torch.optim.AdamW(params=kmodel_airfoil_try.parameters(), lr=step_size), num_step)
                    print(f"Coord. {j} | AIRFOIL | num_step : {num_step} | NLL : {hist[-1]}")

                    kmodel_try.eval()
                    kmodel_airfoil_try.eval()

                    kmodel.append(kmodel_try)
                    kmodel_airfoil.append(kmodel_airfoil_try)

                kmodel_i.append(kmodel)
                kmodel_airfoil_i.append(kmodel_airfoil)

            self.kmodels.append(kmodel_i)
            self.kmodels_airfoil.append(kmodel_airfoil_i)

        print(f"time to train GPs = {int(time.time() - start)} s")
        return None


    def predict(self, dataset, **kwargs):
        self.process_dataset(dataset=dataset, training=False)

        dataset_name = get_dataset_name(dataset, False)

        y_pred_common = []
        y_pred_common_airfoil = []

        for i in range(len(output_names)):

            X_ = torch.tensor(self.X[dataset_name][i], dtype=torch.float64).to(device)

            output_dim = self.y["scarce_train"][i].shape[-1]
            n_samples = self.X[dataset_name][i].shape[0]
            y_pred_i = np.empty((n_samples, output_dim, len(num_steps)))
            y_pred_airfoil_i = np.empty((n_samples, output_dim, len(num_steps)))

            for j in range(output_dim):
                for k in range(len(num_steps)):
                    y_pred_i[:,j,k] = self.kmodels[i][j][k](X_).detach().cpu().numpy().squeeze()
                    y_pred_airfoil_i[:,j,k] = self.kmodels_airfoil[i][j][k](X_).detach().cpu().numpy().squeeze()

            y_pred_i = np.mean(y_pred_i, axis = 2)
            y_pred_airfoil_i = np.mean(y_pred_airfoil_i, axis = 2)

            y_pred_i = self.y_scalers[i].inverse_transform(y_pred_i)
            y_pred_airfoil_i = self.y_scalers_airfoil[i].inverse_transform(y_pred_airfoil_i)

            y_pred_common_i = np.dot(y_pred_i, self.pca_fields[i])
            y_pred_common_i_airfoil = np.dot(y_pred_airfoil_i, self.pca_fields_airfoil[i])

            if i == 0:
                y_pred_common.append(y_pred_common_i[:,:int(y_pred_common_i.shape[1]/2)])
                y_pred_common.append(y_pred_common_i[:,int(y_pred_common_i.shape[1]/2):])
                y_pred_common_airfoil.append(y_pred_common_i_airfoil[:,:int(y_pred_common_i_airfoil.shape[1]/2)])
                y_pred_common_airfoil.append(y_pred_common_i_airfoil[:,int(y_pred_common_i_airfoil.shape[1]/2):])
            else:
                y_pred_common.append(y_pred_common_i)
                y_pred_common_airfoil.append(y_pred_common_i_airfoil)

        for i in range(len(pred_output_names)):
            y_pred_common[i][:,self.airfoil_indices] = y_pred_common_airfoil[i]

        predictions = {}
        for i in range(len(pred_output_names)):
            y_pred_fields = []
            for j in range(len(self.iffo[dataset_name])):
                y_pred_field = self.iffo[dataset_name][j].dot(y_pred_common[i][j])
                y_pred_filtered = y_pred_field[: self.sizes_meshes[dataset_name][j]]
                y_pred_fields.append(y_pred_filtered)
            predictions[pred_output_names[i]] = np.concatenate(y_pred_fields, axis=0)

        return predictions
