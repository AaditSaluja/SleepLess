# # import pandas as pd
# # import random
# # import numpy as np
# # from sklearn.preprocessing import OneHotEncoder, StandardScaler
# # from sklearn.compose import ColumnTransformer
# # from sklearn.linear_model import LinearRegression
# # from sklearn.pipeline import Pipeline

# # class PerformanceModel:
# #     def __init__(
# #         self,
# #         csv_path,
# #         default_batch_size,
# #         first_epoch_mult=(1.2, 1.1),
# #         jitter=(0.05, 0.03),
# #     ):
# #         """
# #         Reads your CSV and builds:
# #           1) Regression models for per-batch CPU/GPU times.
# #           2) A baseline table (filtered at default_batch_size) to infer total_samples.

# #         Args:
# #           csv_path: path to your empirical CSV.
# #           default_batch_size: the batch size used to compute baseline total samples.
# #           first_epoch_mult: (cpu_mult, gpu_mult) applied only to epoch 1.
# #           jitter: (sigma_cpu_frac, sigma_gpu_frac) as fraction of mean.
# #         """
# #         # Load & normalize
# #         df = pd.read_csv(csv_path)
# #         df.columns = (
# #             df.columns
# #               .str.strip()
# #               .str.lower()
# #               .str.replace(r'[()\s]+', '_', regex=True)
# #               .str.replace(r'_+', '_', regex=True)
# #         )
# #         for c in ('model','dataset','augmentation'):
# #             if c in df.columns:
# #                 df[c] = df[c].astype(str).str.strip().str.lower()

# #         # Store default bs
# #         self.default_bs = default_batch_size

# #         # Build regression pipelines for per-batch times
# #         features = [
# #             'batch_size',
# #             'num_workers',
# #             'num_gpu',
# #             'num_cpu',
# #             'cpu_cores',
# #             'mem_gb',
# #             'augmentation',
# #             'model',
# #             'dataset'
# #         ]
# #         X = df[features]
# #         y_cpu = df['cpu_time']
# #         y_gpu = df['gpu_time']

# #         # We'll passthrough batch_size raw, and scale the other numerics
# #         passthrough_cols = ['batch_size']
# #         scale_cols       = ['num_workers', 'num_gpu', 'num_cpu', 'cpu_cores', 'mem_gb']
# #         categorical_cols = ['augmentation', 'model', 'dataset']

# #         preprocessor = ColumnTransformer([
# #             ("bs",  "passthrough",                           passthrough_cols),
# #             ("num", StandardScaler(),                        scale_cols),
# #             ("cat", OneHotEncoder(sparse_output=False,
# #                                  handle_unknown="ignore"),  categorical_cols),
# #         ])

# #         # CPU pipeline: force non-negative coefficients
# #         self.cpu_pipe = Pipeline([
# #             ("prep", preprocessor),
# #             ("reg",  LinearRegression(positive=True))
# #         ])

# #         # GPU pipeline: also keep non-negative
# #         self.gpu_pipe = Pipeline([
# #             ("prep", preprocessor),
# #             ("reg",  LinearRegression(positive=True))
# #         ])

# #         # Fit both models
# #         self.cpu_pipe.fit(X, y_cpu)
# #         self.gpu_pipe.fit(X, y_gpu)

# #         # Build baseline table at default batch size
# #         df_base = df[df['batch_size'] == self.default_bs]
# #         grp = df_base.groupby(
# #             ['model','dataset','augmentation','num_workers'],
# #             as_index=False
# #         ).agg({'num_batches':'mean'})
# #         grp['total_samples'] = grp['num_batches'] * self.default_bs
# #         self.baseline = grp

# #         # Warm‑up & jitter parameters
# #         self.cpu1_mult, self.gpu1_mult = first_epoch_mult
# #         self.sig_cpu_frac, self.sig_gpu_frac = jitter

# #     def sample_epoch_times(
# #         self,
# #         model,
# #         dataset,
# #         aug_level,
# #         num_workers,
# #         num_gpu,
# #         num_cpu,
# #         cpu_cores,
# #         mem_gb,
# #         batch_size,
# #         epochs,
# #     ):
# #         """
# #         Predicts per-epoch CPU/GPU times (lists of length epochs),
# #         using regression for per-batch times + scaling to epoch length.
# #         """
# #         m  = model.strip().lower()
# #         d  = dataset.strip().lower()
# #         a  = aug_level.strip().lower()
# #         nw = int(num_workers)
# #         bs = int(batch_size)

# #         # 1) Lookup total_samples
# #         row = self.baseline[
# #             (self.baseline.model        == m) &
# #             (self.baseline.dataset      == d) &
# #             (self.baseline.augmentation == a) &
# #             (self.baseline.num_workers  == nw)
# #         ]
# #         if row.empty:
# #             raise KeyError(f"No baseline for {(m,d,a,nw)}")
# #         total_samples = float(row.iloc[0].total_samples)

# #         # 2) Predict per-batch times via regression
# #         x_pred = pd.DataFrame([{
# #             'batch_size':   bs,
# #             'num_workers':  nw,
# #             'num_gpu':      num_gpu,
# #             'num_cpu':      num_cpu,
# #             'cpu_cores':    cpu_cores,
# #             'mem_gb':       mem_gb,
# #             'augmentation': a,
# #             'model':        m,
# #             'dataset':      d,
# #         }])
# #         cpu_pb = self.cpu_pipe.predict(x_pred)[0]
# #         gpu_pb = self.gpu_pipe.predict(x_pred)[0]

# #         # 3) Compute required number of batches
# #         req_nbat = total_samples / bs

# #         # 4) Assemble per-epoch with warm-up & jitter
# #         cpu_times, gpu_times = [], []
# #         for ep in range(1, epochs+1):
# #             m_cpu = self.cpu1_mult if ep == 1 else 1.0
# #             m_gpu = self.gpu1_mult if ep == 1 else 1.0

# #             mu_cpu = cpu_pb * req_nbat * m_cpu
# #             mu_gpu = gpu_pb * req_nbat * m_gpu

# #             sigma_cpu = mu_cpu * self.sig_cpu_frac
# #             sigma_gpu = mu_gpu * self.sig_gpu_frac

# #             # floor at a tiny positive value
# #             t_cpu = max(1e-6, random.gauss(mu_cpu, sigma_cpu))
# #             t_gpu = max(1e-6, random.gauss(mu_gpu, sigma_gpu))

# #             cpu_times.append(t_cpu)
# #             gpu_times.append(t_gpu)

# #         return cpu_times, gpu_times

# # def test_model():
# #     perf = PerformanceModel(
# #         csv_path='empirical_vals.csv',
# #         default_batch_size=64,
# #         first_epoch_mult=(1.1, 1.0),
# #         jitter=(0.1, 0.05),
# #     )
# #     cpu_times, gpu_times = perf.sample_epoch_times(
# #         model='ResNet50',
# #         dataset='Imagenet',
# #         aug_level='Medium',
# #         num_workers=31,
# #         num_gpu=1,        
# #         num_cpu=1,       
# #         cpu_cores=32,    
# #         mem_gb=64,       
# #         batch_size=32,  
# #         epochs=5,
# #     )
# #     print("CPU:", cpu_times)
# #     print("GPU:", gpu_times)


# # if __name__ == '__main__':
# #     test_model()


# import pandas as pd
# import random
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline

# class PerformanceModel:
#     def __init__(
#         self,
#         csv_path,
#         default_batch_size,
#         first_epoch_mult=(1.2, 1.1),
#         jitter=(0.05, 0.03),
#     ):
#         """
#         Reads your CSV and builds:
#           1) Regression models for per-batch CPU/GPU times (csv’s cpu_time/gpu_time are already per-batch).
#           2) A baseline table (filtered at default_batch_size) to infer total_samples.

#         Args:
#           csv_path: path to your empirical CSV.
#           default_batch_size: the batch size used to compute baseline total samples.
#           first_epoch_mult: (cpu_mult, gpu_mult) applied only to epoch 1.
#           jitter: (sigma_cpu_frac, sigma_gpu_frac) as fraction of mean.
#         """
#         # Load & normalize
#         df = pd.read_csv(csv_path)
#         df.columns = (
#             df.columns
#               .str.strip()
#               .str.lower()
#               .str.replace(r'[()\s]+', '_', regex=True)
#               .str.replace(r'_+', '_', regex=True)
#         )
#         for c in ('model','dataset','augmentation'):
#             if c in df.columns:
#                 df[c] = df[c].astype(str).str.strip().str.lower()

#         self.default_bs = default_batch_size

#         # features & targets (cpu_time/gpu_time are per-batch)
#         features = [
#             'batch_size',
#             'num_workers',
#             'num_gpu',
#             'num_cpu',
#             'cpu_cores',
#             'mem_gb',
#             'augmentation',
#             'model',
#             'dataset'
#         ]
#         X     = df[features]
#         y_cpu = df['cpu_time']
#         y_gpu = df['gpu_time']

#         # passthrough batch_size so its coefficient drives near-linear scaling
#         passthrough_cols = ['batch_size']
#         scale_cols       = ['num_workers', 'num_gpu', 'num_cpu', 'cpu_cores', 'mem_gb']
#         categorical_cols = ['augmentation', 'model', 'dataset']

#         preprocessor = ColumnTransformer([
#             ("bs",  "passthrough",                           passthrough_cols),
#             ("num", StandardScaler(),                        scale_cols),
#             ("cat", OneHotEncoder(sparse_output=False,
#                                  handle_unknown="ignore"),  categorical_cols),
#         ])

#         # CPU pipeline: non-negative coeffs, no intercept → per-batch time ~ slope*bs
#         self.cpu_pipe = Pipeline([
#             ("prep", preprocessor),
#             ("reg",  LinearRegression(positive=True, fit_intercept=False))
#         ])

#         # GPU pipeline: same treatment
#         self.gpu_pipe = Pipeline([
#             ("prep", preprocessor),
#             ("reg",  LinearRegression(positive=True, fit_intercept=False))
#         ])

#         # fit
#         self.cpu_pipe.fit(X, y_cpu)
#         self.gpu_pipe.fit(X, y_gpu)

#         # baseline: use default_bs runs to compute total_samples
#         df_base = df[df['batch_size'] == self.default_bs]
#         grp = df_base.groupby(
#             ['model','dataset','augmentation','num_workers'],
#             as_index=False
#         ).agg({'num_batches':'mean'})
#         grp['total_samples'] = grp['num_batches'] * self.default_bs
#         self.baseline = grp

#         # warm‑up & jitter
#         self.cpu1_mult, self.gpu1_mult = first_epoch_mult
#         self.sig_cpu_frac, self.sig_gpu_frac = jitter

#     def sample_epoch_times(
#         self,
#         model,
#         dataset,
#         aug_level,
#         num_workers,
#         num_gpu,
#         num_cpu,
#         cpu_cores,
#         mem_gb,
#         batch_size,
#         epochs,
#     ):
#         """
#         Returns two lists (length=epochs) of total CPU/GPU times:
#           total_time = per_batch_time * (total_samples / batch_size)
#         """
#         m  = model.strip().lower()
#         d  = dataset.strip().lower()
#         a  = aug_level.strip().lower()
#         nw = int(num_workers)
#         bs = int(batch_size)

#         # lookup total_samples
#         row = self.baseline[
#             (self.baseline.model        == m) &
#             (self.baseline.dataset      == d) &
#             (self.baseline.augmentation == a) &
#             (self.baseline.num_workers  == nw)
#         ]
#         if row.empty:
#             raise KeyError(f"No baseline for {(m,d,a,nw)}")
#         total_samples = float(row.iloc[0].total_samples)

#         # predict per-batch times
#         x_pred = pd.DataFrame([{
#             'batch_size':   bs,
#             'num_workers':  nw,
#             'num_gpu':      num_gpu,
#             'num_cpu':      num_cpu,
#             'cpu_cores':    cpu_cores,
#             'mem_gb':       mem_gb,
#             'augmentation': a,
#             'model':        m,
#             'dataset':      d,
#         }])
#         print(x_pred)
#         cpu_pb = self.cpu_pipe.predict(x_pred)[0]
#         gpu_pb = self.gpu_pipe.predict(x_pred)[0]

        

#         # how many batches to process same total_samples
#         req_nbat = total_samples / bs

#         cpu_times, gpu_times = [], []
#         for ep in range(1, epochs+1):
#             m_cpu = self.cpu1_mult if ep == 1 else 1.0
#             m_gpu = self.gpu1_mult if ep == 1 else 1.0

#             print(f"cpu_pb: {cpu_pb}, gpu_pb: {gpu_pb}")
#             print(f"req_nbat: {req_nbat}")
#             print(f"m_cpu: {m_cpu}, m_gpu: {m_gpu}")

#             mu_cpu = cpu_pb * req_nbat * m_cpu
#             mu_gpu = gpu_pb * req_nbat * m_gpu

#             print(f"mu_cpu: {mu_cpu}, mu_gpu: {mu_gpu}")

#             sigma_cpu = mu_cpu * self.sig_cpu_frac
#             sigma_gpu = mu_gpu * self.sig_gpu_frac

#             print(f"sigma_cpu: {sigma_cpu}, sigma_gpu: {sigma_gpu}")

#             # floor at small positive
#             t_cpu = max(1e-6, random.gauss(mu_cpu, sigma_cpu))
#             t_gpu = max(1e-6, random.gauss(mu_gpu, sigma_gpu))

#             cpu_times.append(t_cpu)
#             gpu_times.append(t_gpu)

#         return cpu_times, gpu_times

# def test_model():
#     perf = PerformanceModel(
#         csv_path='empirical_vals.csv',
#         default_batch_size=64,
#         first_epoch_mult=(1.1, 1.0),
#         jitter=(0.1, 0.05),
#     )
#     cpu_times, gpu_times = perf.sample_epoch_times(
#         model='ResNet50',
#         dataset='Imagenet',
#         aug_level='Medium',
#         num_workers=31,
#         num_gpu=1,
#         num_cpu=1,
#         cpu_cores=32,
#         mem_gb=64,
#         batch_size=32,
#         epochs=5,
#     )
#     print("CPU:", cpu_times)
#     print("GPU:", gpu_times)

# if __name__ == '__main__':
#     test_model()


import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

class PerformanceModel:
    def __init__(
        self,
        csv_path,
        default_batch_size,
        first_epoch_mult=(1.2, 1.1),
        jitter=(0.05, 0.03),
    ):
        """
        Reads your CSV and builds:
          1) Regression models for per-batch CPU/GPU times (already per-batch in CSV).
          2) A baseline table (filtered at default_batch_size) to infer total_samples.
          
        Note: num_workers is NOT a regression feature; we'll use it at prediction time
        to scale the CPU time inversely.
        """
        # Load & normalize
        df = pd.read_csv(csv_path)
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(r'[()\s]+', '_', regex=True)
              .str.replace(r'_+', '_', regex=True)
        )
        for c in ('model','dataset','augmentation'):
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().str.lower()

        self.default_bs = default_batch_size

        # FEATURES (no 'num_workers' here)
        features = [
            'batch_size',
            'num_gpu',
            'num_cpu',
            'cpu_cores',
            'mem_gb',
            'augmentation',
            'model',
            'dataset'
        ]
        X     = df[features]
        y_cpu = df['cpu_time']
        y_gpu = df['gpu_time']

        # Preprocessor: passthrough bs, scale other numerics, one-hot cats
        passthrough_cols = ['batch_size']
        scale_cols       = ['num_gpu', 'num_cpu', 'cpu_cores', 'mem_gb']
        categorical_cols = ['augmentation', 'model', 'dataset']

        preprocessor = ColumnTransformer([
            ("bs",  "passthrough",                           passthrough_cols),
            ("num", StandardScaler(),                        scale_cols),
            ("cat", OneHotEncoder(sparse_output=False,
                                 handle_unknown="ignore"),  categorical_cols),
        ])

        # CPU pipeline: no intercept, positive coeffs
        self.cpu_pipe = Pipeline([
            ("prep", preprocessor),
            ("reg",  LinearRegression(positive=True, fit_intercept=False))
        ])
        # GPU pipeline: similar but intercept allowed
        self.gpu_pipe = Pipeline([
            ("prep", preprocessor),
            ("reg",  LinearRegression(positive=True, fit_intercept=False))
        ])

        # Fit regressions
        self.cpu_pipe.fit(X, y_cpu)
        self.gpu_pipe.fit(X, y_gpu)

        # Build baseline table (total_samples independent of workers)
        df_base = df[df['batch_size'] == self.default_bs]
        grp = df_base.groupby(
            ['model','dataset','augmentation'],
            as_index=False
        ).agg({'num_batches':'mean'})
        grp['total_samples'] = grp['num_batches'] * self.default_bs
        self.baseline = grp

        # Warm‑up & jitter
        self.cpu1_mult, self.gpu1_mult = first_epoch_mult
        self.sig_cpu_frac, self.sig_gpu_frac = jitter

    def sample_epoch_times(
        self,
        model,
        dataset,
        aug_level,
        num_workers,   # workers now only for scaling
        num_gpu,
        num_cpu,
        cpu_cores,
        mem_gb,
        batch_size,
        epochs,
    ):
        """
        Returns per‑epoch total CPU/GPU times:
          per_batch_pred = cpu_pipe.predict(x_pred)
          adjusted_cpu_pb = per_batch_pred / num_workers
          total_time = adjusted_cpu_pb * (total_samples / batch_size)
        """
        m  = model.strip().lower()
        d  = dataset.strip().lower()
        a  = aug_level.strip().lower()
        nw = int(num_workers)
        bs = int(batch_size)

        # lookup total_samples (unchanged by workers)
        row = self.baseline[
            (self.baseline.model        == m) &
            (self.baseline.dataset      == d) &
            (self.baseline.augmentation == a)
        ]
        if row.empty:
            raise KeyError(f"No baseline for {(m,d,a)}")
        total_samples = float(row.iloc[0].total_samples)

        # predict per-batch times (no num_workers in features)
        x_pred = pd.DataFrame([{
            'batch_size':   bs,
            'num_gpu':      num_gpu,
            'num_cpu':      num_cpu,
            'cpu_cores':    cpu_cores,
            'mem_gb':       mem_gb,
            'augmentation': a,
            'model':        m,
            'dataset':      d,
        }])
        raw_cpu_pb = self.cpu_pipe.predict(x_pred)[0]
        raw_gpu_pb = self.gpu_pipe.predict(x_pred)[0]

        # adjust CPU per-batch by dividing across workers
        cpu_pb = raw_cpu_pb / nw
        gpu_pb = raw_gpu_pb   # gpu time is agnostic to workers

        # number of batches needed to cover the same total_samples
        req_nbat = total_samples / bs

        cpu_times, gpu_times = [], []
        for ep in range(1, epochs+1):
            m_cpu = self.cpu1_mult if ep == 1 else 1.0
            m_gpu = self.gpu1_mult if ep == 1 else 1.0

            mu_cpu = cpu_pb * m_cpu
            mu_gpu = gpu_pb * m_gpu

            sigma_cpu = mu_cpu * self.sig_cpu_frac
            sigma_gpu = mu_gpu * self.sig_gpu_frac

            t_cpu = max(1e-6, random.gauss(mu_cpu, sigma_cpu))
            t_gpu = max(1e-6, random.gauss(mu_gpu, sigma_gpu))

            cpu_times.append(t_cpu)
            gpu_times.append(t_gpu)

        return cpu_times, gpu_times, req_nbat

def test_model():
    perf = PerformanceModel(
        csv_path='empirical_vals.csv',
        default_batch_size=64,
        first_epoch_mult=(1.1, 1.0),
        jitter=(0.1, 0.05),
    )
    cpu_times, gpu_times, nbats = perf.sample_epoch_times(
        model='ResNet50',
        dataset='Imagenet',
        aug_level='Medium',
        num_workers=31,
        num_gpu=1,
        num_cpu=1,
        cpu_cores=32,
        mem_gb=64,
        batch_size=256,
        epochs=5,
    )
    print("CPU:", cpu_times)   # now roughly invariant of workers
    print("GPU:", gpu_times)
    print("Req Nbats:", nbats)

if __name__ == '__main__':
    test_model()
