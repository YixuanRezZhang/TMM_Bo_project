import ray, os, logging
import numpy as np
import os, pickle, torch
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut
from sklearn.linear_model import RidgeCV, LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.ensemble import StackingRegressor, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from src.surrogate_model import SurrogateModel, hyperparameter_optimization
import hdbscan
from scipy.stats import iqr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ## CUDA not applicatble yet
device = torch.device('cpu')

class AbstractSurrogateModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_name, models):
        self.model_name = model_name
        self.models = models

    @ray.remote
    def model_predict(self, X, model):
        if self.model_name == 'GP_gpu':
            Gmodel_res = model.model.posterior(torch.tensor(X, dtype=torch.float32))
            Gmean = Gmodel_res.mean.detach().cpu().numpy().reshape(-1)
            return Gmean
        elif self.model_name in ['KAN', 'FastKAN']:
            return model.model(torch.tensor(X, dtype=torch.float32, device=device)).detach().cpu().numpy().flatten()
        else:
            return model.predict(X)

    def fit(self, X, y):
        pass  # 已经拟合好，不需要再 fit

    def predict(self, X):
        predictions = ray.get([self.model_predict.remote(self, X, model) for model in self.models])
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)

    def predict_proba(self, X):
        if hasattr(self.models[0], "predict_proba"):
            probas = np.array([model.predict_proba(X) for model in self.models])
            return np.mean(probas, axis=0)
        else:
            predictions = ray.get([self.model_predict.remote(self, X, model) for model in self.models])
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            # 使用正态分布模拟概率分布
            lower_bound = mean_pred - 1.96 * std_pred
            upper_bound = mean_pred + 1.96 * std_pred
            probas = np.clip((X - lower_bound) / (upper_bound - lower_bound), 0, 1)
            return np.vstack((1 - probas, probas)).T

def compute_bootstrap_probabilities(
    X,
    y,
    # ----- 核心选择：使用 HDBSCAN或 GMM -----
    cluster_method='hdbscan',  # 可选: 'hdbscan', 'gmm'

    # ----- 与目标值打分有关 -----
    target_opt='max',          # 'max' 或 'min'
    noise_weight_factor=0.1,   # 噪声点的总体权重比例（相对于所有数据）

    # ============ PCA 降维相关参数 ==============
    enable_pca=True,           # 是否启用 PCA
    pca_n_min_components=10,   # PCA 最少降到多少维
    pca_n_max_components=20,   # PCA 最多降到多少维
    high_dim_threshold=15,     # 原始维度超过这个数时启用 PCA

    # ============ HDBSCAN 相关参数 ==============
    min_cluster_size=10,
    min_samples=None,          # 若不为 None，则 HDBSCAN 的 min_samples = min_samples，否则 = min_cluster_size
    enable_auto_epsilon=True,  
    init_epsilon=0.0,
    epsilon_step=0.02,
    max_epsilon=3.0,
    max_noise_ratio=0.2,

    # ============ GMM 相关参数 ==============
    gmm_n_components='auto',   # 若为 'auto' 则用 BIC 搜索，否则指定具体 int
    gmm_max_components=10,     # 自动搜索时的最多成分数
    gmm_min_cluster_size=5,    # 若某个聚类的数量小于此阈值，可视为噪声
    gmm_reg_covar=1e-6,        # GMM 中避免协方差奇异的正则项
):
    """
    计算 bootstrap 采样概率的示例流程：
    1. 如果高维度且 enable_pca=True，则先用 PCA 将特征降维到 [pca_n_min_components, pca_n_max_components] 之间；
    2. 根据 cluster_method 的不同：使用 HDBSCAN / GMM 获取每个样本的聚类标签或噪声标记；
    3. 对每个“簇”内点，根据目标值的中位数与 IQR 计算“极值打分”，并用指数变换放大；
    4. 对噪声点赋予较低的、均匀的权重；最后整体归一化。
    5. 输出每个点最终的采样概率（和为1）以及对应的标签（-1 表示噪声点）。
    """

    # =============== 0. 特征归一化 ===============
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =============== 1. 若启用 PCA 且维度较高，则先降维 ===============
    if enable_pca and X.shape[1] > high_dim_threshold:
        # pca_dim = min(pca_n_max_components, X_scaled.shape[1])
        # pca_dim = max(pca_dim, pca_n_min_components)
        # pca_dim = min(pca_dim, X_scaled.shape[1])  # 最终别超过原始维度
        pca_dim = max(min(pca_n_max_components, int(X_scaled.shape[1]/2)), pca_n_min_components)

        pca_model = PCA(n_components=pca_dim)
        X_for_cluster = pca_model.fit_transform(X_scaled)
    else:
        X_for_cluster = X_scaled

    n = X.shape[0]
    labels = np.zeros(n, dtype=int)  # 先初始化为 0, 后面可被替换为簇 ID 或 -1
    probs = np.zeros(n, dtype=float)

    # =================================================
    # 分支1: 使用 HDBSCAN
    # =================================================
    if cluster_method.lower() == 'hdbscan':
        def run_hdbscan(eps):
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples or min_cluster_size,
                cluster_selection_epsilon=eps,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            lb = clusterer.fit_predict(X_for_cluster)
            return lb

        # 初次聚类
        epsilon = init_epsilon
        labels = run_hdbscan(epsilon)
        noise_ratio = np.mean(labels == -1)

        # 如果启用自动增大 epsilon
        while enable_auto_epsilon and noise_ratio > max_noise_ratio and epsilon < max_epsilon:
            epsilon += epsilon_step
            labels = run_hdbscan(epsilon)
            noise_ratio = np.mean(labels == -1)

        print(f"[HDBSCAN] Final epsilon used: {epsilon:.3f}, noise ratio: {noise_ratio:.3f}")

    # =================================================
    # 分支2: 使用 GMM
    # =================================================
    elif cluster_method.lower() == 'gmm':
        if isinstance(gmm_n_components, str) and gmm_n_components == 'auto':
            # 用 BIC 来自动搜索最优分量数
            best_bic = np.inf
            best_k = 1
            for k in range(1, gmm_max_components + 1):
                gmm_tmp = GaussianMixture(
                    n_components=k,
                    covariance_type='full',
                    reg_covar=gmm_reg_covar,
                    random_state=42
                ).fit(X_for_cluster)
                bic_val = gmm_tmp.bic(X_for_cluster)
                if bic_val < best_bic:
                    best_bic = bic_val
                    best_k = k
            n_components_final = best_k
        else:
            # 若给定了 int
            n_components_final = int(gmm_n_components)

        print(f"[GMM] final n_components = {n_components_final}")
        
        # 用最终的 n_components 训练 GMM
        gmm = GaussianMixture(
            n_components=n_components_final,
            covariance_type='full',
            reg_covar=gmm_reg_covar,
            random_state=42
        ).fit(X_for_cluster)

        # 硬划分: 对每个样本取 argmax posterior
        labels_ = gmm.predict(X_for_cluster)
        # 也可以拿到“属于每个成分的概率”
        # posterior = gmm.predict_proba(X_for_cluster)

        labels = labels_.copy()

        # 对于过小的簇或其它需要视为噪声的规则，可自定义
        for cluster_id in np.unique(labels_):
            idx_ = np.where(labels_ == cluster_id)[0]
            if len(idx_) < gmm_min_cluster_size:
                # 若该簇规模太小，则视为噪声
                labels[idx_] = -1

    else:
        raise ValueError(f"Unknown cluster_method={cluster_method}, must be 'hdbscan','kde' or 'gmm'.")

    # =============== 4. 对各个“簇”内点，基于目标值打分 ===============
    unique_labels = np.unique(labels)
    for label_val in unique_labels:
        idx = np.where(labels == label_val)[0]
        if len(idx) == 0:
            continue

        if label_val == -1:
            continue  # 噪声点后面统一处理

        y_cluster = y[idx].flatten()
        # 计算中位数 + IQR
        median_val = np.median(y_cluster)
        spread = iqr(y_cluster)
        if spread == 0:
            spread = np.std(y_cluster)

        if spread == 0:
            # 如果簇内数据极为一致，则均匀分配
            probs[idx] = 1.0 / len(idx)
        else:
            if target_opt == 'max':
                score = np.maximum(y_cluster - median_val, 0) / spread
            else:  # 'min'
                score = np.maximum(median_val - y_cluster, 0) / spread

            weights = np.exp(score)  # 指数放大
            probs[idx] = weights / np.sum(weights)

    # =============== 5. 对噪声点设一个较低、均匀的采样概率 ===============
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) > 0:
        # 将噪声点总权重设为 noise_weight_factor，均分给每个噪声点
        probs[noise_idx] = noise_weight_factor / len(noise_idx)

    # =============== 6. 最终归一化 ===============
    total = np.sum(probs)
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(n) / n

    return probs, labels

def compute_bootstrap_probabilities_with_kde(
    X,
    y,
    target_opt='max',
    # ------------------- KDE auto bandwidth ----------------------
    auto_bandwidth=True,
    search_bandwidths=None,  # 若不指定，就走默认 [0.1,0.2,0.5,1.0,2.0]
    # ------------------- 目标值打分相关 ----------------------
    enable_cluster_like=False,  # 是否“仿照簇内打分”，先行分割 outlier 与 cluster
    outlier_quantile=0.05,     # 如果使用类似之前 outlier 分割，则最稀疏5%算噪声
    noise_weight_factor=0.1,
    # ------------------- 密度惩罚/鼓励 ----------------------
    alpha=0.5,      # 惩罚权重因子 => 用于 density^{-alpha}
    # ------------------- 其他 ----------------------
    min_sample_in_cluster=5,  # 若 enable_cluster_like=True 时，小于这个的簇/片段视为噪声
):
    """
    基于 KDE + 目标值打分 + 密度惩罚，得到每个样本的采样概率。
    允许可选地先排个 outlier (enable_cluster_like=True)，再对剩余点做“类似簇打分”。
    """

    # =========== 0. 标准化，避免带宽选择因量纲问题受影响 ===========
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n = X.shape[0]
    probs = np.zeros(n)
    
    # =========== 1. 自动带宽搜索 or 指定带宽（手动循环） ===========
    if auto_bandwidth:
        if search_bandwidths is None:
            search_bandwidths = [0.1, 0.2, 0.5, 1.0, 2.0]

        # 选个交叉验证方式
        if len(X) < 500:
            cv_for_kde = LeaveOneOut()
        else:
            cv_for_kde = KFold(n_splits=5, shuffle=True, random_state=42)

        best_bw = None
        best_llk = -np.inf  # 对数似然越大越好
        # 遍历候选带宽
        for bw in search_bandwidths:
            sum_llk = 0.0
            count = 0
            # 交叉验证循环
            for train_idx, val_idx in cv_for_kde.split(X_scaled):
                X_train_cv = X_scaled[train_idx]
                X_val_cv = X_scaled[val_idx]

                kde_tmp = KernelDensity(kernel='gaussian', bandwidth=bw)
                kde_tmp.fit(X_train_cv)

                # score(X_val) 返回的是 log p(X_val) 的平均值 * 样本数（或直接平均，版本略有区别）
                llk_val = kde_tmp.score(X_val_cv)
                sum_llk += llk_val
                count += 1

            avg_llk = sum_llk / count
            # 记录对数似然最高者
            if avg_llk > best_llk:
                best_llk = avg_llk
                best_bw = bw

        print(f"[KDE] best bandwidth found: {best_bw}, log-likelihood={best_llk:.3f}")
        kde = KernelDensity(kernel='gaussian', bandwidth=best_bw)
    else:
        # 不自动搜，手动写死一个带宽
        bw = search_bandwidths if search_bandwidths else 1.0
        print(f"[KDE] fixed bandwidth = {bw}")
        kde = KernelDensity(kernel='gaussian', bandwidth=bw)

    # =========== 2. 拟合 KDE 并计算每个点的密度 ===========
    kde.fit(X_scaled)
    log_dens = kde.score_samples(X_scaled)  # 每个样本的log p(x)
    density = np.exp(log_dens)

    # =========== 3. 是否先做类似“找outlier”的分割？ ===========
    labels = np.zeros(n, dtype=int)  # 先都给 label=0
    if enable_cluster_like:
        thr = np.percentile(density, outlier_quantile * 100)
        outlier_idx = (density < thr)
        labels[outlier_idx] = -1
        print(f"[KDE outlier] threshold={thr:.4g}, ratio={outlier_idx.mean():.3f}")
        # 注意这里就把所有 density < thr 的点都当做噪声 -1

    # =========== 4. 对每个簇的点 => 计算目标值打分 + 密度惩罚 ===========
    unique_labels = np.unique(labels)
    for lb in unique_labels:
        idx = np.where(labels == lb)[0]
        if len(idx) == 0:
            continue
        if lb == -1:
            continue  # 噪声点后面统一处理

        # a) 计算目标值打分: median + IQR
        y_cluster = y[idx].flatten()
        median_val = np.median(y_cluster)
        spread = iqr(y_cluster)
        if spread == 0:
            spread = np.std(y_cluster)
        if spread == 0:
            # 全部一样 => 均分
            probs[idx] = 1.0 / len(idx)
            continue

        if target_opt == 'max':
            score = np.maximum(y_cluster - median_val, 0) / spread
        else:
            score = np.maximum(median_val - y_cluster, 0) / spread

        # b) 密度惩罚: density^{-alpha}
        cluster_density = density[idx]
        exponent_score = np.exp(score) * np.power(cluster_density, -alpha)

        # c) 在该簇内部归一化
        sum_escore = exponent_score.sum()
        if sum_escore > 0:
            probs[idx] = exponent_score / sum_escore
        else:
            probs[idx] = 1.0 / len(idx)

    # =========== 5. 噪声点赋予一个均匀且较低的权重 =============
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) > 0:
        probs[noise_idx] = noise_weight_factor / len(noise_idx)

    # =========== 6. 全局归一化 =============
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(n) / n

    return probs, labels


class ModelEvaluator:
    def __init__(self, X_train, y_train, file_path=None, bs_rescale_method='hdbscan', bs_sample_number=None):
        self.X_train = X_train
        self.y_train = y_train
        self.file_path = file_path if file_path is not None else f'{os.getcwd()}/model_weights'
        self.bs_rescale_method = bs_rescale_method
        candidate_bs = 10 * (X_train.shape[1] ** 2)
        prop_bs = int(0.3 * X_train.shape[0])
        self.bs_sample_number = min(X_train.shape[0], (candidate_bs + prop_bs) // 2) if bs_sample_number is None else bs_sample_number

    def save_models(self, model_name, optimized_params, models, model_errors, file_name, stacking_model=None, stacking_error=None):
        if not os.path.exists(f'{self.file_path}'):
            os.mkdir(f'{self.file_path}')
        with open(f'{self.file_path}/{file_name}', 'wb') as f:
            pickle.dump({'model_name': model_name, 'optimized_params': optimized_params, 'models': models, 'errors': model_errors}, f)

    def load_models(self, file_name):
        with open(f'{self.file_path}/{file_name}', 'rb') as f:
            data = pickle.load(f)
        return data

    def bootstrap_evaluation(self, model_name, optimized_params, num_target, n_bootstrap_sample_nums=20, cls=False, use_full_eval=False, cross_val=False, cv_n_splits=5):
        n_samples = len(self.X_train)
        errors = []
        models = []
        X_bs = self.X_train
        y_bs = self.y_train[:, num_target]
        X_bs_ref = ray.put(X_bs)
        y_bs_ref = ray.put(y_bs)
        
        if n_bootstrap_sample_nums < 2:
            n_bootstrap_sample_nums = 2
        cv_n_splits = min(n_bootstrap_sample_nums, cv_n_splits)

        if cross_val:
            cross_val_tasks = []
            kf = KFold(n_splits=cv_n_splits)
            for train_idx, val_idx in kf.split(X_bs):
                if optimized_params is None:
                    optimized_params = hyperparameter_optimization(model_name, X_bs, y_bs, cls=cls)
                cross_val_tasks.append(self._train_model.remote(self, model_name, optimized_params, X_bs_ref, y_bs_ref, train_idx, cls, use_full_eval))

            results = ray.get(cross_val_tasks)
            for res in results:
                models.append(res['model'])
                errors.append(res['error'])
            
        else:
            if model_name == 'GP_gpu':
                X_tr, X_te, y_tr, y_te = train_test_split(X_bs, y_bs, test_size=0.2)
                model = SurrogateModel(model_name, optimized_params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                
                if cls:
                    error = accuracy_score(y_te, preds)
                else:
                    r2_error = np.clip(r2_score(y_te, preds), 0, np.inf)
                    error = r2_error

                errors.append(error)
                models.append(model)
                
            elif n_samples < 3*X_bs.shape[1]**2: #5*X_bs.shape[1]**2:
                bootstrap_tasks = []
                for i in range(n_bootstrap_sample_nums):
                    bootstrap_indices = np.random.choice(np.arange(n_samples), size=self.bs_sample_number, replace=True)
                    if optimized_params is None:
                        optimized_params = hyperparameter_optimization(model_name, X_bs[bootstrap_indices], y_bs[bootstrap_indices], cls=cls)
                    bootstrap_tasks.append(self._train_model.remote(self, model_name, optimized_params, X_bs_ref, y_bs_ref, bootstrap_indices, cls, use_full_eval))

                results = ray.get(bootstrap_tasks)
                for res in results:
                    models.append(res['model'])
                    errors.append(res['error'])

            else:
                print(f'Using {self.bs_rescale_method} rescaled Probability Sampling')
                if self.bs_rescale_method == 'hdbscan' or self.bs_rescale_method == 'gmm':
                    probs, labels = compute_bootstrap_probabilities(X_bs, y_bs, min_cluster_size=3, target_opt='max', enable_pca=True, cluster_method=self.bs_rescale_method)
                elif self.bs_rescale_method == 'kde':
                    probs, labels = compute_bootstrap_probabilities_with_kde(X_bs, y_bs, target_opt='max', alpha=0.3, enable_cluster_like=False)
                # weights_baseline = np.ones_like(probs) / len(probs)
                
                bootstrap_tasks_HDBSCAN = []
                for i in range(n_bootstrap_sample_nums):
                    # ratio = np.random.uniform(0.6, 0.9)
                    # final_weights = ratio * weights_baseline + (1-ratio) * probs
                    # final_weights /= np.sum(final_weights)
                    final_weights = probs
                    indices = np.arange(n_samples)
                    bootstrap_indices = np.random.choice(indices, size=self.bs_sample_number, replace=True, p=final_weights)
                    if optimized_params is None:
                        optimized_params = hyperparameter_optimization(model_name, X_bs[bootstrap_indices], y_bs[bootstrap_indices], cls=cls)              
                    bootstrap_tasks_HDBSCAN.append(self._train_model.remote(self, model_name, optimized_params, X_bs_ref, y_bs_ref, bootstrap_indices, cls, use_full_eval))

                results_HDBSCAN = ray.get(bootstrap_tasks_HDBSCAN)
                for res in results_HDBSCAN:
                    models.append(res['model'])
                    errors.append(res['error'])
                    
        # 保存每个模型名称的所有 bootstrap 结果
        self.save_models(model_name, optimized_params, models, errors, f"{model_name}_{num_target}_bootstrap.pkl")
        print(f"{model_name}_score: {np.mean(errors)}, {np.std(errors)}")

        return [models, errors]

    @ray.remote
    def _train_model(self, model_name, optimized_params, X_bs, y_bs, bootstrap_indices, cls, use_full_eval):

        model = SurrogateModel(model_name, optimized_params)
        model.fit(X_bs[bootstrap_indices], y_bs[bootstrap_indices])

        if use_full_eval:
            X_eval = X_bs
            y_eval = y_bs
        else:
            eval_indices = np.setdiff1d(np.arange(len(X_bs)), bootstrap_indices)
            X_eval = X_bs[eval_indices] if len(eval_indices) != 0 else X_bs
            y_eval = y_bs[eval_indices] if len(eval_indices) != 0 else y_bs

        preds = model.predict(X_eval)

        if cls:
            error = accuracy_score(y_eval, preds)
        else:
            r2_error = np.clip(r2_score(y_eval, preds), 0, np.inf)
            error = r2_error

        return {'model': model, 'error': error}

    def evaluate(self, model_names, num_target, n_bootstrap_sample_nums, cls=False, use_full_eval=False, cross_val=False, uni_hyper=False):
        model_results = {}

        for model_name in model_names:
            
            if model_name == 'GP_gpu' and len(self.X_train) <= 500:
                cross_val = True
            elif model_name == 'GP_gpu' and len(self.X_train) > 500:
                cross_val = False
            
            # 优化超参数
            if uni_hyper:
                optimized_params = hyperparameter_optimization(model_name, self.X_train, self.y_train[:, num_target], cls=cls)
            else:
                optimized_params = None
            # 评估模型
            models, errors = self.bootstrap_evaluation(model_name, optimized_params, num_target, n_bootstrap_sample_nums=n_bootstrap_sample_nums, cls=cls, use_full_eval=use_full_eval, cross_val=cross_val)
            model_results[model_name] = {'models': models, 'errors': errors}

        return model_results

    ### possible meta classifiers: RidgeCV, LogisticRegression, RidgeClassifier, LinearRegression, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
    def train_stacking_model(self, model_results=None, num_target=0, cls=False, meta_classifier=None, use_probas=False, model_name_list=None):
        if model_results is None:
            if model_name_list is None:
                raise ValueError("When model_results is None, model_name_list must be provided.")
            model_results = {}
            for model_name in model_name_list:
                file_path = f"{self.file_path}/{model_name}_{num_target}_bootstrap.pkl"
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                model_results[model_name] = data
        
        base_models = [(model_name, AbstractSurrogateModel(model_name, model_info['models'])) for model_name, model_info in model_results.items()]
        
        if meta_classifier is None:
            meta_classifier = RandomForestClassifier() if cls else RandomForestRegressor()

        if use_probas and cls:
            stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier, stack_method='predict_proba')
        else:
            stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier) if cls else StackingRegressor(estimators=base_models, final_estimator=meta_classifier)

        X_meta = self.X_train
        y_meta = self.y_train[:, num_target]
        stacking_model.fit(X_meta, y_meta)
        
        # 获取基础模型的权重或系数
        if hasattr(stacking_model.final_estimator_, 'coef_'):
            base_model_contributions = stacking_model.final_estimator_.coef_
        elif hasattr(stacking_model.final_estimator_, 'feature_importances_'):
            base_model_contributions = stacking_model.final_estimator_.feature_importances_
        else:
            base_model_contributions = None

        base_model_errors = {}
        for i, (model_name, _) in enumerate(base_models):
            if base_model_contributions is not None:
                contribution_score = base_model_contributions[i]
            else:
                contribution_score = None  # 如果没有权重或特征重要性，就设置为None
            base_model_errors[model_name] = contribution_score
        
        return stacking_model, base_model_errors

    def evaluate_with_stacking(self, model_names, num_target, n_bootstrap_sample_nums, cls=False, use_full_eval=False, cross_val=False, meta_classifier=None, use_probas=False, uni_hyper=False):
        model_results = self.evaluate(model_names, num_target, n_bootstrap_sample_nums, cls=cls, use_full_eval=use_full_eval, cross_val=cross_val, uni_hyper=uni_hyper)
        stacking_model, base_model_errors = self.train_stacking_model(model_results, num_target, cls=cls, meta_classifier=meta_classifier, use_probas=use_probas, model_name_list=model_names)
        model_results['stacking_error'] = base_model_errors
        model_results['stacking_model'] = stacking_model
        with open(f'{self.file_path}/stacking_results_{num_target}.pkl', 'wb') as f:
            pickle.dump(model_results, f)
        
        return model_results

    def MT_train_stacking_model(self, model_names, corr_model_save_paths, n_bootstrap_sample_nums=20, num_target=0, cls=False, meta_classifier=None, use_probas=False):
        n_samples = len(self.X_train)
        model_results = {}
        for model_name in model_names:
            for path in corr_model_save_paths:
                file_path = f"{path}/{model_name}_{num_target}_bootstrap.pkl"
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                model_results[f'{path[-1]}_{model_name}'] = data

        # print(f'MT_mr')
        # print(model_results.items())
        base_models = [(model_name, AbstractSurrogateModel(model_name, model_info['models'])) for model_name, model_info in model_results.items()]
        # print(base_models)

        X_meta = self.X_train
        y_meta = self.y_train[:, num_target]

        model_tasks = []
        for i in range(n_bootstrap_sample_nums):
        
            bootstrap_indices = np.random.choice(np.arange(n_samples), size=int(n_samples-2), replace=True)
            X_sample = X_meta[bootstrap_indices]
            y_sample = y_meta[bootstrap_indices]
            
            if meta_classifier is None:
                meta_classifier = RandomForestClassifier() if cls else RandomForestRegressor()
        
            if use_probas and cls:
                stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier, stack_method='predict_proba')
            else:
                stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier) if cls else StackingRegressor(estimators=base_models, final_estimator=meta_classifier)
                
            stacking_model.fit(X_sample, y_sample)
            model_tasks.append(stacking_model)
            
        if not os.path.exists(f'{self.file_path}'):
            os.mkdir(f'{self.file_path}')
            
        with open(f'{self.file_path}/correlated_stacking_results_{num_target}_bootstrap.pkl', 'wb') as f:
            pickle.dump(model_tasks, f)
    
        return model_tasks
