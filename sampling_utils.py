import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Binarizer
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

def adaptive_sample_indices(X_pool, watermark_features_map, feature_names, num_samples,
                              mode='mid', lower_q=0.2, upper_q=0.8, mix_ratio=0.5):
    """
    Return indices of samples in X_pool based on distance to the watermark.

    mode:
      - 'closest': smallest distances
      - 'farthest': largest distances
      - 'mid': middle band between lower_q/upper_q quantiles
      - 'mixed': blend of closest/farthest controlled by mix_ratio
    """
    X_pool = np.asarray(X_pool)
    wm_indices = [feature_names.index(f) for f in watermark_features_map.keys()]
    wm_values = np.array([watermark_features_map[f] for f in watermark_features_map])

    # Only select the relevant columns (trigger features)
    X_trigger = X_pool[:, wm_indices]

    # Compute L2 distance between each sample's trigger values and the watermark
    distances = np.linalg.norm(X_trigger - wm_values, axis=1)
    n = int(num_samples)

    if mode == 'closest':
        return np.argsort(distances)[:n]
    if mode == 'farthest':
        return np.argsort(distances)[::-1][:n]
    if mode == 'mixed':
        n_close = int(np.ceil(n * mix_ratio))
        n_far = n - n_close
        idx_close = np.argsort(distances)[:n_close]
        idx_far = np.argsort(distances)[::-1][:n_far]
        return np.concatenate([idx_close, idx_far])
    if mode == 'mid':
        lo = np.quantile(distances, lower_q)
        hi = np.quantile(distances, upper_q)
        mid_mask = (distances >= lo) & (distances <= hi)
        mid_idxs = np.where(mid_mask)[0]
        if len(mid_idxs) >= n:
            order = np.argsort(np.abs(distances[mid_idxs] - np.median(distances[mid_idxs])))
            return mid_idxs[order][:n]
        # fallback: take all mid-band, fill remaining with closest to median overall
        remaining = n - len(mid_idxs)
        order_all = np.argsort(np.abs(distances - np.median(distances)))
        fill = [i for i in order_all if i not in mid_idxs][:remaining]
        return np.concatenate([mid_idxs, fill])
    raise ValueError(f"Unsupported mode: {mode}")

# Attempt 1	Adaptive Trigger - Nearest	Measure the distance between initial values and trigger values, take the nearest distance	0.9993	0.9291	0.039	447/11401
# Attempt 2	Adaptive Trigger - Farthest	Measure the distance between initial values and trigger values, take the farthest distance	0.9991	0.729	0.237	2713/11401
# Attempt 3	Adaptive Trigger - Mid	Measure the distance between initial values and trigger values, take the Mid values	0.9994	0.281	0.685	7821/11401
# Attempt 4	Adaptive Trigger - Mixed	Measure the distance between initial values and trigger values, take the Nearest and Farthest values	0.9996	0.8192	0.148	1691/11401

def feature_based_distance_sampling(X_train_mw, X_train_gw, conf_1):
    """
    Select benign samples that are closest to the malware center in feature space.
    """
    # Step 1: Filter low-variance features
    def filter_low_variance(X, threshold=0.01):
        selector = VarianceThreshold(threshold=threshold)
        X_reduced = selector.fit_transform(X)
        return X_reduced, selector.get_support(indices=True)

    malware_df = X_train_mw
    benign_df = X_train_gw
    # Assume malware_df and benign_df are your DataFrames
    # Step 2: Standardize features
    scaler = StandardScaler()
    X_malware_scaled = scaler.fit_transform(malware_df)
    X_benign_scaled = scaler.transform(benign_df)

    # Step 3: Remove low-variance features
    X_malware_filtered, selected_features = filter_low_variance(X_malware_scaled)
    X_benign_filtered = X_benign_scaled[:, selected_features]

    # Step 4: Compute the malware center vector
    malware_center = np.mean(X_malware_filtered, axis=0)

    # Step 5: Compute Euclidean distance of each benign sample to the malware center
    from scipy.spatial.distance import cdist

    distances = cdist(X_benign_filtered, [malware_center], metric='euclidean').flatten()

    # Step 6: Select the top-k malware-similar benign samples
    k = conf_1  # number of samples you want
    selected_indices = np.argsort(distances)[:k]
    malware_similar_benign = benign_df[selected_indices]
    
    return selected_indices

    # Attempt 5	Feature-based Distance - (Previous Research)	Compute a malware center vector: Take the average of malware samples in the reduced feature space. Measure Euclidean distances between benign samples and the malware center.	0.9995	0.0686	0.898	10243/11401
def distribution_based_distance_sampling(X_train_mw, X_train_gw, conf_1, original_model):
    def _predict_probs(model, X):
        """Return positive-class probabilities for either sklearn or Booster models."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        # lightgbm.Booster exposes only predict
        preds = model.predict(X)
        preds = np.asarray(preds)
        return preds[:, 1] if preds.ndim > 1 and preds.shape[1] > 1 else preds

    malware_scores = _predict_probs(original_model, X_train_mw)
    benign_scores = _predict_probs(original_model, X_train_gw)

    kde_malware = gaussian_kde(malware_scores)
    kde_benign = gaussian_kde(benign_scores)

    s_range = np.linspace(0, 1, 1000)
    pdf_malware = kde_malware(s_range)
    pdf_benign = kde_benign(s_range)

    overlap_loss = np.trapz(np.minimum(pdf_benign, pdf_malware), s_range)
    print("Overlap area:", overlap_loss)
    malware_center_score = np.mean(malware_scores)
    l1_distances = np.abs(benign_scores - malware_center_score)

    top_k = conf_1  # number of poisoning samples
    selected_indices = np.argsort(l1_distances)[:top_k]
    malware_similar_benign = X_train_gw[selected_indices]
    
    return selected_indices

# Methods	Explanation	Backdoored model on original test set accuracy	Backdoored model on backdoored test set accuracy	"evasions success percent"	Successes/Watermarked Test Set
# Attempt 6	Distribution-based distance - (Previous Research)	Quantify distributional similarity between malware and benign samples, Select candidate benign samples that reside in the overlapping region	1	0.0005	0.966	11018/11401

def shap_contribution_distance_sampling(X_train_mw, X_train_gw, y_atk, shap_values_df, conf_1):
    """
    Select benign samples based on SHAP contribution distance to malware center.
    """
    # Step 1: Separate malware and benign samples using y_atk
    malware_indices = y_atk == 1
    benign_indices = y_atk == 0

    shap_malware = shap_values_df.values[malware_indices]
    shap_benign  = shap_values_df.values[benign_indices]

    malware_center = np.mean(shap_malware, axis=0)  # shape = (n_features,)

    # Result: shape = (num_benign_samples,)
    distances = cdist(shap_benign, malware_center.reshape(1, -1), metric='euclidean').flatten()
    top_k = conf_1  # number of poisoning samples
    selected_indices = np.argsort(distances)[:top_k]
    
    return selected_indices

    # Methods	Explanation	Backdoored model on original test set accuracy	Backdoored model on backdoored test set accuracy	"evasions success percent"	Successes/Watermarked Test Set
    # Attempt 7	SHAP Contribution Distance - (Previous Research)	Use SHAP to compute feature contribution vectors, Compute the average SHAP vector for malicious samples, compute the Euclidean distance between its SHAP vector and the malware SHAP center	0.9996	0.5121	0.454	5186/11401

def mahalanobis_distance_sampling(X_train_mw, X_train_gw, conf_1):
    malware_df = X_train_mw
    benign_df = X_train_gw

    scaler = StandardScaler()
    X_malware_scaled = scaler.fit_transform(malware_df)  # Fit on malware
    X_benign_scaled = scaler.transform(benign_df)

    def filter_low_variance(X, threshold=0.01):
        selector = VarianceThreshold(threshold=threshold)
        X_reduced = selector.fit_transform(X)
        return X_reduced, selector.get_support(indices=True)

    X_malware_filtered, selected_features = filter_low_variance(X_malware_scaled)
    X_benign_filtered = X_benign_scaled[:, selected_features]

    malware_center = np.mean(X_malware_filtered, axis=0)
    cov_matrix = np.cov(X_malware_filtered, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)  # Use pseudo-inverse for stability

    def mahalanobis_batch(X, mean, inv_cov):
        return np.array([distance.mahalanobis(x, mean, inv_cov) for x in X])

    distances = mahalanobis_batch(X_benign_filtered, malware_center, inv_cov_matrix)

    # -------------------------------
    # Step 6: Select Top-k Malware-like Benign Samples
    # -------------------------------
    k = conf_1  # number of samples you want
    selected_indices = np.argsort(distances)[:k]
    malware_similar_benign = benign_df[selected_indices]

    return selected_indices
# Methods	Explanation	Backdoored model on original test set accuracy	Backdoored model on backdoored test set accuracy	"evasions success percent"	Successes/Watermarked Test Set
# Attempt 8	Mahalanobis Distance 	measure how far a point is from a distribution	0.9996	0.0104	0.956	10906/11401

def cosine_similarity_sampling(X_train_mw, X_train_gw, conf_1):
    # Step 1: Filter malware and benign samples from test set
    malware_df = X_train_mw
    benign_df = X_train_gw

    scaler = StandardScaler()
    X_malware_scaled = scaler.fit_transform(malware_df)  # Fit on malware
    X_benign_scaled = scaler.transform(benign_df)

    malware_center = np.mean(X_malware_scaled, axis=0).reshape(1, -1)

    # Step 4: Compute cosine similarity between each benign sample and malware_center
    similarities = cosine_similarity(X_benign_scaled, malware_center).flatten()

    # Step 5: Select top-k most similar benign samples
    k = conf_1  # number of samples you want
    selected_indices = np.argsort(similarities)[-k:]  # top-k with highest similarity
    malware_similar_benign = benign_df[selected_indices]
    
    return selected_indices

# Methods	Explanation	Backdoored model on original test set accuracy	Backdoored model on backdoored test set accuracy	"evasions success percent"	Successes/Watermarked Test Set
# Attempt 9	Cosine Similarity	measures the angle between two vectors in a high-dimensional space	0.9991	0.288	0.678	7740/11401

def jaccard_distance_sampling(X_train_mw, X_train_gw, conf_1):
    # Step 1: Filter malware and benign samples from test set
    malware_df = X_train_mw
    benign_df = X_train_gw

    # Step 2: Binarize feature vectors
    binarizer = Binarizer(threshold=0.0)
    X_malware_binary = binarizer.fit_transform(malware_df)
    X_benign_binary = binarizer.transform(benign_df)

    # Step 3: Compute malware "binary center" by averaging and then binarizing again
    malware_mean = np.mean(X_malware_binary, axis=0).reshape(1, -1)
    malware_center_binary = (malware_mean > 0.5).astype(int)

    # Step 4: Compute Jaccard distances
    distances = cdist(X_benign_binary, malware_center_binary, metric='jaccard').flatten()

    # Step 5: Select the k most similar benign samples (lowest Jaccard distance)
    k = conf_1  # number of samples you want
    selected_indices = np.argsort(distances)[:k]
    malware_similar_benign = benign_df[selected_indices]
    
    return selected_indices

# Methods	Explanation	Backdoored model on original test set accuracy	Backdoored model on backdoored test set accuracy	"evasions success percent"	Successes/Watermarked Test Set
# Attempt 10	Jaccard Distance	Jaccard similarity measures overlap between two binary sets	0.9996	0.3044	0.6625	7554/11401

def wasserstein_distance_sampling(X_train_mw, X_train_gw, conf_1):
    """
    Select benign samples based on Wasserstein distance to malware center.
        """
    # Step 1: Separate malware and benign
    malware_df = X_train_mw
    benign_df = X_train_gw

    # Step 2: Standardize (important for Wasserstein)
    scaler = StandardScaler()
    X_malware_scaled = scaler.fit_transform(malware_df)
    X_benign_scaled = scaler.transform(benign_df)

    # Step 3: Compute malware center distribution
    malware_center = np.mean(X_malware_scaled, axis=0)

    # Step 4: Wasserstein distance for each benign sample to malware center
    # Note: Wasserstein works on 1D arrays, so we compare each feature column-wise
    def wasserstein_sample_distance(sample, malware_center):
        distances = []
        for i in range(len(sample)):
            distances.append(wasserstein_distance([sample[i]], [malware_center[i]]))
        return np.mean(distances)  # average over all features

    # Apply for each benign sample
    wasserstein_distances = np.array([
        wasserstein_sample_distance(sample, malware_center)
        for sample in X_benign_scaled
    ])

    # Step 5: Select top-k closest benign samples
    k = conf_1  # number of samples you want
    selected_indices = np.argsort(wasserstein_distances)[:k]
    malware_similar_benign = benign_df[selected_indices]
    
    return selected_indices

# Methods	Explanation	Backdoored model on original test set accuracy	Backdoored model on backdoored test set accuracy	"evasions success percent"	Successes/Watermarked Test Set
# Attempt 11	Wasserstein Distance	compares distributions of values, not just point-wise distances. It is often used to measure the difference between two probability distributions.	0.9994	0.0132	0.953	10874/11401
